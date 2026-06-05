from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .de_encoder import DE_Encoder
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
        self.hcfb = HCFB(
            shallow_channels=dims[0],
            deep_channels=dims[2],
            task_dim=task_dim,
            shallow_heads=4,
            deep_heads=8,
        )
        self.mid_fusion = nn.Sequential(
            nn.Conv2d(dims[1] * 2, dims[1], kernel_size=1, bias=False),
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

    def encode_modalities(self, img_ir, img_vis):
        ir_feats = self.encoder_ir(img_ir)
        vis_feats = self.encoder_vis(img_vis)
        return ir_feats, vis_feats

    def build_mid_features(self, ir_feats, vis_feats):
        return self.mid_fusion(torch.cat([ir_feats["stage2"], vis_feats["stage2"]], dim=1))

    def fuse_features(self, ir_feats, vis_feats, mid_fused):
        fusion = self.hcfb(
            ir_feats["stage1"],
            vis_feats["stage1"],
            ir_feats["stage3"],
            vis_feats["stage3"],
            task_signals=None,
        )
        return {
            "shallow": fusion["fused_shallow"],
            "mid": mid_fused,
            "deep": fusion["fused_deep"],
            "stats": fusion["stats"],
        }

    def decode(self, fused_features, img_ir, img_vis_rgb):
        residual = 0.5 * (img_ir + img_vis_rgb.mean(dim=1, keepdim=True))
        return self.decoder(
            fused_features["deep"],
            fused_features["mid"],
            fused_features["shallow"],
            residual=residual,
        )

    def apply_objectness_guidance(self, fused, task_signals):
        if task_signals is None or "obj" not in task_signals:
            return fused
        obj = task_signals["obj"].detach()
        obj = F.interpolate(obj, size=fused.shape[-2:], mode="bilinear", align_corners=False)
        obj = obj.clamp(0.0, 1.0)
        detail = fused - F.avg_pool2d(fused, kernel_size=3, stride=1, padding=1)
        guided = fused + self.objectness_guidance_alpha * obj * detail
        return guided.clamp(0.0, 1.0)

    def forward(
        self,
        img_ir,
        img_vis,
        targets: Optional[List[Dict[str, torch.Tensor]]] = None,
        run_detection: bool = True,
    ):
        ir_feats, vis_feats = self.encode_modalities(img_ir, img_vis)
        base_mid = self.build_mid_features(ir_feats, vis_feats)
        fused_features = self.fuse_features(ir_feats, vis_feats, base_mid)
        fused_output = self.decode(fused_features, img_ir, img_vis)

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
