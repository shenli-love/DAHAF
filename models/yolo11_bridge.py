from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics import YOLO
from ultralytics.nn.tasks import DetectionModel


class YOLO11Detector(nn.Module):
    def __init__(
        self,
        cfg="yolo11n.yaml",
        num_classes=6,
        ch=3,
        image_size=256,
        box_gain=7.5,
        cls_gain=0.5,
        dfl_gain=1.5,
        pretrained_weights=None,
        freeze=True,
        verbose=False,
    ):
        super().__init__()
        self.model = DetectionModel(cfg, ch=ch, nc=num_classes, verbose=verbose)
        self.model.args = SimpleNamespace(box=box_gain, cls=cls_gain, dfl=dfl_gain)
        self.num_classes = num_classes
        self.image_size = image_size
        self.frozen = freeze
        self.pretrained_loaded = False

        if pretrained_weights:
            self._load_pretrained(pretrained_weights, verbose=verbose)
        if self.frozen:
            self.freeze()

        with torch.no_grad():
            sample = torch.zeros(1, ch, image_size, image_size)
            outputs = self.model(sample)
            parsed = outputs[1] if isinstance(outputs, tuple) else outputs
        self.feature_channels = [feat.shape[1] for feat in parsed["feats"]]

    def _load_pretrained(self, weights, verbose=False):
        try:
            yolo = YOLO(weights)
            self.model.load(yolo.model, verbose=verbose)
            self.pretrained_loaded = True
        except Exception as exc:
            self.pretrained_loaded = False
            if verbose:
                print(f"[YOLO11Detector] failed to load pretrained weights from {weights}: {exc}")

    def freeze(self):
        for param in self.model.parameters():
            param.requires_grad_(False)
        self.model.eval()

    def forward(self, image):
        if image.shape[1] == 1:
            image = image.repeat(1, 3, 1, 1)
        elif image.shape[1] != 3:
            raise ValueError("YOLO11Detector expects 1-channel or 3-channel input.")
        preds = self.model(image)
        parsed = preds[1] if isinstance(preds, tuple) else preds
        return {
            "img": image,
            "preds": preds,
            "boxes": parsed["boxes"],
            "scores": parsed["scores"],
            "feats": parsed["feats"],
        }

    def build_batch(self, image, targets):
        device = image.device
        height, width = image.shape[-2:]
        batch_idx = []
        cls = []
        bboxes = []
        for batch_id, target in enumerate(targets):
            boxes = target["boxes"].to(device)
            labels = target["labels"].to(device)
            if boxes.numel() == 0:
                continue
            xywh = boxes.clone()
            xywh[:, 0] = (boxes[:, 0] + boxes[:, 2]) * 0.5 / width
            xywh[:, 1] = (boxes[:, 1] + boxes[:, 3]) * 0.5 / height
            xywh[:, 2] = (boxes[:, 2] - boxes[:, 0]) / width
            xywh[:, 3] = (boxes[:, 3] - boxes[:, 1]) / height
            xywh = xywh.clamp_(0.0, 1.0)

            batch_idx.append(torch.full((boxes.shape[0],), batch_id, dtype=torch.long, device=device))
            cls.append(labels.float().unsqueeze(1))
            bboxes.append(xywh)

        if batch_idx:
            batch_idx = torch.cat(batch_idx, dim=0)
            cls = torch.cat(cls, dim=0)
            bboxes = torch.cat(bboxes, dim=0)
        else:
            batch_idx = torch.zeros((0,), dtype=torch.long, device=device)
            cls = torch.zeros((0, 1), dtype=torch.float32, device=device)
            bboxes = torch.zeros((0, 4), dtype=torch.float32, device=device)

        return {
            "img": image,
            "batch_idx": batch_idx,
            "cls": cls,
            "bboxes": bboxes,
        }

    def loss(self, detector_outputs, targets):
        batch = self.build_batch(detector_outputs["img"], targets)
        loss_items, detached = self.model.loss(batch, detector_outputs["preds"])
        total = loss_items.sum() if loss_items.ndim > 0 else loss_items
        metrics = {
            "box": detached[0].detach(),
            "cls": detached[1].detach(),
            "dfl": detached[2].detach(),
        }
        return total, metrics


class TaskBridge(nn.Module):
    def __init__(self, detector_channels, fusion_dim=64, bridge_channels=128):
        super().__init__()
        self.projections = nn.ModuleList(
            [nn.Conv2d(ch, bridge_channels, kernel_size=1, bias=False) for ch in detector_channels]
        )
        self.proj_norms = nn.ModuleList([nn.BatchNorm2d(bridge_channels) for _ in detector_channels])
        self.fusion = nn.Sequential(
            nn.Conv2d(bridge_channels * len(detector_channels), bridge_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(bridge_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(bridge_channels, bridge_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(bridge_channels),
            nn.SiLU(inplace=True),
        )
        self.object_head = nn.Conv2d(bridge_channels, 1, kernel_size=1)

    def forward(self, det_feats, ir_feats, vis_feats, mid_fused):
        deep_size = ir_feats["stage3"].shape[-2:]
        mid_size = mid_fused.shape[-2:]

        projected = []
        for feat, proj, norm in zip(det_feats, self.projections, self.proj_norms):
            feat = F.silu(norm(proj(feat)))
            projected.append(F.interpolate(feat, size=deep_size, mode="bilinear", align_corners=False))

        bridge_feat = self.fusion(torch.cat(projected, dim=1))
        object_map = torch.sigmoid(self.object_head(bridge_feat))
        object_mid = F.interpolate(object_map, size=mid_size, mode="bilinear", align_corners=False)

        return {
            "mid": mid_fused,
            "obj": object_mid,
            "bridge_feat": bridge_feat,
        }
