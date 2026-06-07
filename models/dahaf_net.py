from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .de_encoder import DE_Encoder, make_norm
from .decoder import DE_Decoder
from .hcfb import HCFB
from .yolo11_bridge import TaskBridge, YOLO11Detector


class DAHAFNet(nn.Module):
    """Single-path DAHAF-Net Lite."""

    def __init__(
        self,
        inp_channels=1,
        dims=(64, 128, 256),
        num_classes=6,
        detector_cfg="yolo11n.yaml",
        detector_input_size=256,
        detector_box_gain=7.5,
        detector_cls_gain=0.5,
        detector_dfl_gain=1.5,
        detector_pretrained="yolo11n.pt",
        detector_freeze=True,
        detector_initial_no_grad=True,
        task_dim=64,
        bridge_channels=128,
        objectness_guidance_alpha=0.05,
    ):
        super().__init__()
        self.objectness_guidance_alpha = objectness_guidance_alpha
        self.encoder_ir = DE_Encoder(inp_channels=inp_channels, dims=dims)
        self.encoder_vis = DE_Encoder(inp_channels=inp_channels, dims=dims)
        self.vis_chroma_adapter = nn.ModuleDict({
            "stage1": self._make_chroma_adapter(2, dims[0]),
            "stage2": self._make_chroma_adapter(2, dims[1]),
            "stage3": self._make_chroma_adapter(2, dims[2]),
        })
        self.vis_chroma_gate = nn.ModuleDict({
            "stage1": self._make_chroma_gate(dims[0]),
            "stage3": self._make_chroma_gate(dims[2]),
        })
        self.vis_detail_adapter = nn.ModuleDict({
            "stage1": self._make_chroma_adapter(1, dims[0]),
            "stage2": self._make_chroma_adapter(1, dims[1]),
            "stage3": self._make_chroma_adapter(1, dims[2]),
        })
        self.vis_detail_gate = nn.ModuleDict({
            "stage1": self._make_chroma_gate(dims[0]),
            "stage3": self._make_chroma_gate(dims[2]),
        })
        self.hcfb = HCFB(
            shallow_channels=dims[0],
            deep_channels=dims[2],
            task_dim=task_dim,
            shallow_heads=4,
            deep_heads=8,
        )
        self.mid_fusion = nn.Sequential(
            nn.Conv2d(dims[1] * 3, dims[1], kernel_size=1, bias=False),
            nn.BatchNorm2d(dims[1]),
            nn.SiLU(inplace=True),
        )
        self.decoder = DE_Decoder(
            deep_dim=dims[2],
            mid_dim=dims[1],
            shallow_dim=dims[0],
            out_channels=1,
        )
        self.detector = YOLO11Detector(
            cfg=detector_cfg,
            num_classes=num_classes,
            image_size=detector_input_size,
            box_gain=detector_box_gain,
            cls_gain=detector_cls_gain,
            dfl_gain=detector_dfl_gain,
            pretrained_weights=detector_pretrained,
            freeze=detector_freeze,
            verbose=False,
        )
        self.task_bridge = TaskBridge(
            detector_channels=self.detector.feature_channels,
            fusion_dim=dims[1],
            bridge_channels=bridge_channels,
        )
        self.detector_initial_no_grad = detector_initial_no_grad

    @staticmethod
    def _make_chroma_adapter(in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            make_norm(out_channels),
            nn.SiLU(inplace=True),
        )

    @staticmethod
    def _make_chroma_gate(channels):
        return nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

    def _extract_vis_chroma_features(self, vis_feats, img_vis_chroma):
        if img_vis_chroma is None:
            return None
        if img_vis_chroma.shape[1] != 2:
            raise ValueError(f"Expected img_vis_chroma with 2 channels, got {img_vis_chroma.shape[1]}")

        chroma = img_vis_chroma.to(device=vis_feats["stage1"].device, dtype=vis_feats["stage1"].dtype)
        chroma_feats = {}
        for stage_name, adapter in self.vis_chroma_adapter.items():
            resized_chroma = F.interpolate(
                chroma,
                size=vis_feats[stage_name].shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            chroma_feats[stage_name] = adapter(resized_chroma)
        return chroma_feats

    def _extract_vis_detail_features(self, vis_feats, img_vis_detail):
        if img_vis_detail is None:
            return None
        if img_vis_detail.shape[1] != 1:
            raise ValueError(f"Expected img_vis_detail with 1 channel, got {img_vis_detail.shape[1]}")

        detail = img_vis_detail.to(device=vis_feats["stage1"].device, dtype=vis_feats["stage1"].dtype)
        detail_feats = {}
        for stage_name, adapter in self.vis_detail_adapter.items():
            resized_detail = F.interpolate(
                detail,
                size=vis_feats[stage_name].shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            detail_feats[stage_name] = adapter(resized_detail)
        return detail_feats

    def encode_modalities(self, img_ir, img_vis, img_vis_chroma=None, detail_map=None):
        ir_feats = self.encoder_ir(img_ir)
        vis_feats = self.encoder_vis(img_vis)
        chroma_feats = self._extract_vis_chroma_features(vis_feats, img_vis_chroma)
        detail_feats = self._extract_vis_detail_features(vis_feats, detail_map)
        return ir_feats, vis_feats, chroma_feats, detail_feats

    def build_mid_features(self, ir_feats, vis_feats, chroma_feats=None, detail_feats=None):
        chroma_mid = (
            chroma_feats["stage2"]
            if chroma_feats is not None
            else torch.zeros_like(vis_feats["stage2"])
        )
        detail_mid = detail_feats["stage2"] if detail_feats is not None else 0.0
        return self.mid_fusion(torch.cat([ir_feats["stage2"], vis_feats["stage2"], chroma_mid + detail_mid], dim=1))

    def _enhance_vis_with_aux(self, vis_feat, chroma_feat, detail_feat, stage_name):
        aux_feat = None
        if chroma_feat is not None:
            aux_feat = chroma_feat
        if detail_feat is not None:
            detail_gate = self.vis_detail_gate[stage_name](detail_feat)
            aux_feat = detail_gate * detail_feat if aux_feat is None else aux_feat + detail_gate * detail_feat
        if aux_feat is None:
            return vis_feat
        gate = self.vis_chroma_gate[stage_name](aux_feat)
        return vis_feat + gate * aux_feat

    def fuse_features(self, ir_feats, vis_feats, mid_fused, chroma_feats=None, detail_feats=None, task_signals=None):
        chroma_stage1 = chroma_feats["stage1"] if chroma_feats is not None else None
        chroma_stage3 = chroma_feats["stage3"] if chroma_feats is not None else None
        detail_stage1 = detail_feats["stage1"] if detail_feats is not None else None
        detail_stage3 = detail_feats["stage3"] if detail_feats is not None else None
        vis_stage1 = self._enhance_vis_with_aux(vis_feats["stage1"], chroma_stage1, detail_stage1, "stage1")
        vis_stage3 = self._enhance_vis_with_aux(vis_feats["stage3"], chroma_stage3, detail_stage3, "stage3")
        fusion = self.hcfb(
            ir_feats["stage1"],
            vis_stage1,
            ir_feats["stage3"],
            vis_stage3,
            task_signals=task_signals,
            shallow_detail=detail_stage1,
            deep_detail=detail_stage3,
        )
        return {
            "shallow": fusion["fused_shallow"],
            "mid": mid_fused,
            "deep": fusion["fused_deep"],
            "stats": fusion["stats"],
        }

    def decode(self, fused_features, img_ir, img_vis, detail_map=None):
        vis_blur = F.avg_pool2d(img_vis, kernel_size=5, stride=1, padding=2)
        residual = img_vis - vis_blur
        return self.decoder(
            fused_features["deep"],
            fused_features["mid"],
            fused_features["shallow"],
            residual=residual,
            detail_map=detail_map,
        )

    def apply_objectness_guidance(self, fused, task_signals):
        if task_signals is None or "obj" not in task_signals:
            return fused
        obj = task_signals["obj"]
        obj = F.interpolate(obj, size=fused.shape[-2:], mode="bilinear", align_corners=False)
        obj = obj.clamp(0.0, 1.0)
        detail = fused - F.avg_pool2d(fused, kernel_size=3, stride=1, padding=1)
        guided = fused + self.objectness_guidance_alpha * obj * detail
        return guided.clamp(0.0, 1.0)

    def forward(
        self,
        img_ir,
        img_vis,
        img_vis_chroma: Optional[torch.Tensor] = None,
        detail_map: Optional[torch.Tensor] = None,
        targets: Optional[List[Dict[str, torch.Tensor]]] = None,
        run_detection: bool = True,
    ):
        ir_feats, vis_feats, chroma_feats, detail_feats = self.encode_modalities(
            img_ir,
            img_vis,
            img_vis_chroma=img_vis_chroma,
            detail_map=detail_map,
        )
        base_mid = self.build_mid_features(
            ir_feats,
            vis_feats,
            chroma_feats=chroma_feats,
            detail_feats=detail_feats,
        )
        fused_features = self.fuse_features(
            ir_feats,
            vis_feats,
            base_mid,
            chroma_feats=chroma_feats,
            detail_feats=detail_feats,
        )
        fused_output = self.decode(fused_features, img_ir, img_vis, detail_map=detail_map)

        detector_initial = None
        task_signals = None

        if run_detection:
            if self.detector_initial_no_grad:
                with torch.no_grad():
                    detector_initial = self.detector(fused_output)
            else:
                detector_initial = self.detector(fused_output)

            task_signals = self.task_bridge(
                detector_initial["feats"], ir_feats, vis_feats, base_mid
            )
            fused_features = self.fuse_features(
                ir_feats,
                vis_feats,
                base_mid,
                chroma_feats=chroma_feats,
                detail_feats=detail_feats,
                task_signals=task_signals,
            )
            fused_output = self.decode(fused_features, img_ir, img_vis, detail_map=detail_map)
            fused_output = self.apply_objectness_guidance(fused_output, task_signals)

        return {
            "fused": fused_output,
            "detector_initial": detector_initial,
            "task_signals": task_signals,
            "fusion_stats": fused_features["stats"],
            "ir_feats": ir_feats,
            "vis_feats": vis_feats,
        }

    def detection_loss(self, outputs, targets):
        if outputs["detector_initial"] is None:
            zero = outputs["fused"].new_tensor(0.0)
            metrics = {
                "box": zero,
                "cls": zero,
                "dfl": zero,
            }
            return zero, metrics

        loss_init, metrics_init = self.detector.loss(outputs["detector_initial"], targets)
        metrics = {
            "box": metrics_init["box"],
            "cls": metrics_init["cls"],
            "dfl": metrics_init["dfl"],
        }
        return loss_init, metrics
