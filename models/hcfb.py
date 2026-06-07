import torch
import torch.nn as nn
import torch.nn.functional as F


class MCE(nn.Module):
    """Modality confidence evaluator."""

    def __init__(self, channels, hidden_ratio=4, confidence_floor=0.15):
        super().__init__()
        hidden = max(channels // hidden_ratio, 8)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.confidence_floor = confidence_floor
        self.mlp = nn.Sequential(
            nn.Linear(channels * 2, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 2),
            nn.Sigmoid(),
        )

    def forward(self, f_ir, f_vis):
        descriptor = torch.cat(
            [
                self.pool(f_ir).flatten(1),
                self.pool(f_vis).flatten(1),
            ],
            dim=1,
        )
        weights = self.mlp(descriptor)
        weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-6)
        if self.confidence_floor > 0.0:
            weights = weights * (1.0 - 2.0 * self.confidence_floor) + self.confidence_floor
        w_ir = weights[:, 0:1].unsqueeze(-1).unsqueeze(-1)
        w_vis = weights[:, 1:2].unsqueeze(-1).unsqueeze(-1)
        return w_ir, w_vis


class CrossGatedLocalMixer(nn.Module):
    def __init__(self, channels):
        super().__init__()
        hidden = max(channels // 4, 8)
        self.ir_local = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
        )
        self.vis_local = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
        )
        self.spatial_gate = nn.Sequential(
            nn.Conv2d(channels * 3 + 1, hidden, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, 2, kernel_size=1, bias=True),
        )
        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels * 2, hidden, kernel_size=1, bias=False),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, channels * 2, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )
        self.gate_ir = nn.Parameter(torch.tensor(0.0))
        self.gate_vis = nn.Parameter(torch.tensor(0.0))

    def forward(self, f_ir, f_vis, objectness=None):
        ir_local = self.ir_local(f_ir)
        vis_local = self.vis_local(f_vis)
        discrepancy = (f_ir - f_vis).abs()

        if objectness is None:
            objectness = f_ir.new_zeros(f_ir.shape[0], 1, f_ir.shape[2], f_ir.shape[3])
        else:
            objectness = F.interpolate(objectness, size=f_ir.shape[-2:], mode="bilinear", align_corners=False)

        spatial_logits = self.spatial_gate(torch.cat([ir_local, vis_local, discrepancy, objectness], dim=1))
        spatial_ir, spatial_vis = torch.chunk(spatial_logits.softmax(dim=1), 2, dim=1)

        channel_weights = self.channel_gate(torch.cat([ir_local, vis_local], dim=1))
        channel_ir, channel_vis = torch.chunk(channel_weights, 2, dim=1)

        mixed_ir = spatial_vis * vis_local * channel_vis
        mixed_vis = spatial_ir * ir_local * channel_ir

        f_ir = f_ir + torch.sigmoid(self.gate_ir) * mixed_ir
        f_vis = f_vis + torch.sigmoid(self.gate_vis) * mixed_vis
        return f_ir, f_vis


class TaskAwareDetailInjection(nn.Module):
    def __init__(self, channels):
        super().__init__()
        hidden = max(channels // 4, 8)
        self.gate = nn.Sequential(
            nn.Conv2d(channels * 2 + 1, hidden, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )
        self.scale = nn.Parameter(torch.tensor(0.2))

    def forward(self, f_vis, detail_feat=None, objectness=None):
        if detail_feat is None:
            return f_vis, None

        detail_feat = F.interpolate(detail_feat, size=f_vis.shape[-2:], mode="bilinear", align_corners=False)
        if objectness is None:
            objectness = f_vis.new_zeros(f_vis.shape[0], 1, f_vis.shape[2], f_vis.shape[3])
        else:
            objectness = F.interpolate(objectness, size=f_vis.shape[-2:], mode="bilinear", align_corners=False)
            objectness = objectness.clamp(0.0, 1.0)

        gate = self.gate(torch.cat([f_vis, detail_feat, objectness], dim=1))
        task_weight = 0.25 + 0.75 * objectness
        injected = f_vis + torch.sigmoid(self.scale) * task_weight * gate * detail_feat
        return injected, {
            "gate": gate.detach(),
            "objectness": objectness.detach(),
        }


class SpatialBaseDetailFusion(nn.Module):
    def __init__(self, channels):
        super().__init__()
        hidden = max(channels // 4, 8)
        self.router_temperature = 0.9
        self.base_extractor = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=5, padding=2, groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
        )
        self.detail_extractor = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
        )
        self.edge_proj = nn.Sequential(
            nn.Conv2d(2, hidden, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
        )
        self.router = nn.Sequential(
            nn.Conv2d(channels * 2 + hidden + 1, hidden, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, 4, kernel_size=1, bias=True),
        )
        self.detail_scale = nn.Parameter(torch.tensor(1.10))
        self.edge_scale = nn.Parameter(torch.tensor(0.12))

    def _split(self, feat):
        base = self.base_extractor(feat)
        detail = self.detail_extractor(feat - base)
        return base, detail

    def _edge_strength(self, feat):
        grad_x = feat[:, :, :, 1:] - feat[:, :, :, :-1]
        grad_x = F.pad(grad_x.abs(), (0, 1, 0, 0))
        grad_y = feat[:, :, 1:, :] - feat[:, :, :-1, :]
        grad_y = F.pad(grad_y.abs(), (0, 0, 0, 1))
        return (grad_x + grad_y).mean(dim=1, keepdim=True)

    def forward(self, f_ir, f_vis, objectness=None):
        base_ir, detail_ir = self._split(f_ir)
        base_vis, detail_vis = self._split(f_vis)
        edge_ir = self._edge_strength(detail_ir)
        edge_vis = self._edge_strength(detail_vis)
        edge_feat = self.edge_proj(torch.cat([edge_ir, edge_vis], dim=1))

        if objectness is None:
            objectness = f_ir.new_zeros(f_ir.shape[0], 1, f_ir.shape[2], f_ir.shape[3])
        else:
            objectness = F.interpolate(objectness, size=f_ir.shape[-2:], mode="bilinear", align_corners=False)

        weights = self.router(torch.cat([f_ir, f_vis, edge_feat, objectness], dim=1))
        base_logits, detail_logits = torch.chunk(weights, 2, dim=1)
        base_weights = F.softmax(base_logits / self.router_temperature, dim=1)
        detail_weights = F.softmax(detail_logits / self.router_temperature, dim=1)
        w_ir_base, w_vis_base = torch.chunk(base_weights, 2, dim=1)
        w_ir_detail, w_vis_detail = torch.chunk(detail_weights, 2, dim=1)

        fused_base = w_ir_base * base_ir + w_vis_base * base_vis
        fused_detail = w_ir_detail * detail_ir + w_vis_detail * detail_vis
        edge_focus = 0.5 * (edge_ir * detail_ir + edge_vis * detail_vis)
        fused = fused_base + self.detail_scale * fused_detail + self.edge_scale * edge_focus
        return fused, torch.cat([base_weights, detail_weights], dim=1)


class CrossLayerTransfer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.align = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
        )
        self.gate = nn.Parameter(torch.tensor(-2.0))

    def forward(self, shallow, deep):
        aligned = F.adaptive_avg_pool2d(self.align(shallow), deep.shape[-2:])
        transfer = 0.3 + 0.4 * torch.sigmoid(self.gate)
        return deep + transfer * aligned


class HCFBLevel(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.mce = MCE(channels)
        self.detail_inject = TaskAwareDetailInjection(channels)
        self.mixer = CrossGatedLocalMixer(channels)
        self.fusion = SpatialBaseDetailFusion(channels)

    def forward(self, f_ir, f_vis, objectness=None, detail_feat=None):
        f_vis, detail_stats = self.detail_inject(f_vis, detail_feat=detail_feat, objectness=objectness)
        w_ir, w_vis = self.mce(f_ir, f_vis)
        f_ir, f_vis = self.mixer(f_ir * w_ir, f_vis * w_vis, objectness=objectness)
        fused, router_weights = self.fusion(f_ir, f_vis, objectness=objectness)
        return fused, {
            "w_ir": w_ir,
            "w_vis": w_vis,
            "router": router_weights,
            "detail_injection": detail_stats,
        }


class HCFB(nn.Module):
    """Lite hierarchical fusion core."""

    def __init__(self, shallow_channels=64, deep_channels=256, task_dim=64, shallow_heads=4, deep_heads=8):
        super().__init__()
        self.level1 = HCFBLevel(shallow_channels)
        self.level2 = HCFBLevel(deep_channels)
        self.transfer = CrossLayerTransfer(shallow_channels, deep_channels)

    def forward(
        self,
        shallow_ir,
        shallow_vis,
        deep_ir,
        deep_vis,
        task_signals=None,
        shallow_detail=None,
        deep_detail=None,
    ):
        objectness = None if task_signals is None else task_signals["obj"]
        fused_shallow, shallow_stats = self.level1(
            shallow_ir,
            shallow_vis,
            objectness=objectness,
            detail_feat=shallow_detail,
        )
        deep_ir = self.transfer(fused_shallow, deep_ir)
        deep_vis = self.transfer(fused_shallow, deep_vis)
        deep_objectness = None if objectness is None else F.interpolate(objectness, size=deep_ir.shape[-2:], mode="bilinear", align_corners=False)
        fused_deep, deep_stats = self.level2(
            deep_ir,
            deep_vis,
            objectness=deep_objectness,
            detail_feat=deep_detail,
        )

        return {
            "fused_shallow": fused_shallow,
            "fused_deep": fused_deep,
            "stats": {
                "shallow": shallow_stats,
                "deep": deep_stats,
            },
        }
