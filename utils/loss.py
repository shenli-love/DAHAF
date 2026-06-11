import torch
import torch.nn as nn
import torch.nn.functional as F


class SobelXY(nn.Module):
    def __init__(self):
        super().__init__()
        kernel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        kernel_y = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer("weight_x", kernel_x)
        self.register_buffer("weight_y", kernel_y)

    def forward(self, x):
        grad_x = F.conv2d(x, self.weight_x, padding=1)
        grad_y = F.conv2d(x, self.weight_y, padding=1)
        return grad_x.abs() + grad_y.abs()


class Laplacian(nn.Module):
    def __init__(self):
        super().__init__()
        kernel = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer("weight", kernel)

    def forward(self, x):
        return F.conv2d(x, self.weight, padding=1).abs()


class SSIMLoss(nn.Module):
    def __init__(self, window_size=11):
        super().__init__()
        self.window_size = window_size

    def forward(self, x, y):
        c1 = 0.01 ** 2
        c2 = 0.03 ** 2
        mu_x = F.avg_pool2d(x, self.window_size, stride=1, padding=self.window_size // 2)
        mu_y = F.avg_pool2d(y, self.window_size, stride=1, padding=self.window_size // 2)
        sigma_x = F.avg_pool2d(x * x, self.window_size, stride=1, padding=self.window_size // 2) - mu_x.pow(2)
        sigma_y = F.avg_pool2d(y * y, self.window_size, stride=1, padding=self.window_size // 2) - mu_y.pow(2)
        sigma_xy = F.avg_pool2d(x * y, self.window_size, stride=1, padding=self.window_size // 2) - mu_x * mu_y
        ssim_map = ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / (
            (mu_x.pow(2) + mu_y.pow(2) + c1) * (sigma_x + sigma_y + c2)
        )
        return 1.0 - ssim_map.mean()


class FusionLoss(nn.Module):
    def __init__(
        self,
        gradient_weight=5.0,
        edge_weight=2.0,
        ssim_weight=1.0,
        objectness_weight=0.4,
        detail_weight=1.0,
        semantic_weight=0.2,
        local_window_size=7,
    ):
        super().__init__()
        self.sobel = SobelXY()
        self.laplacian = Laplacian()
        self.ssim = SSIMLoss()
        self.gradient_weight = gradient_weight
        self.edge_weight = edge_weight
        self.ssim_weight = ssim_weight
        self.objectness_weight = objectness_weight
        self.detail_weight = detail_weight
        self.semantic_weight = semantic_weight
        self.local_window_size = local_window_size

    @staticmethod
    def _set_requires_grad(module, requires_grad):
        states = []
        for param in module.parameters():
            states.append((param, param.requires_grad))
            param.requires_grad_(requires_grad)
        return states

    @staticmethod
    def _restore_requires_grad(states):
        for param, requires_grad in states:
            param.requires_grad_(requires_grad)

    @staticmethod
    def _normalized_feature_mse(feat, target):
        feat = F.normalize(feat, dim=1, eps=1e-6)
        target = F.normalize(target.detach(), dim=1, eps=1e-6)
        return F.mse_loss(feat, target)

    def _local_variance(self, x):
        kernel = self.local_window_size
        mean = F.avg_pool2d(x, kernel, stride=1, padding=kernel // 2)
        mean_sq = F.avg_pool2d(x * x, kernel, stride=1, padding=kernel // 2)
        return (mean_sq - mean.pow(2)).clamp_min(0.0)

    def _semantic_consistency_loss(self, img_fused, img_ir, img_vis, semantic_detector):
        if semantic_detector is None:
            return img_fused.new_zeros(())

        was_training = semantic_detector.training
        grad_states = self._set_requires_grad(semantic_detector, False)
        semantic_detector.eval()
        try:
            fused_feats = semantic_detector(img_fused)["feats"]
            with torch.no_grad():
                ir_feats = semantic_detector(img_ir)["feats"]
                vis_feats = semantic_detector(img_vis)["feats"]
        finally:
            self._restore_requires_grad(grad_states)
            semantic_detector.train(was_training)

        loss = img_fused.new_zeros(())
        for fused_feat, ir_feat, vis_feat in zip(fused_feats, ir_feats, vis_feats):
            loss = loss + 0.5 * (
                self._normalized_feature_mse(fused_feat, ir_feat)
                + self._normalized_feature_mse(fused_feat, vis_feat)
            )
        return loss / max(len(fused_feats), 1)

    def forward(
        self,
        img_fused,
        img_ir,
        img_vis,
        objectness_map=None,
        detail_map=None,
        semantic_detector=None,
    ):
        var_ir = self._local_variance(img_ir)
        var_vis = self._local_variance(img_vis)
        contrast_sum = var_ir + var_vis
        contrast_conf = contrast_sum / (contrast_sum + 0.01)
        weight_ir = 0.5 * (1.0 - contrast_conf) + contrast_conf * (var_ir / (contrast_sum + 1e-6))
        weight_vis = 1.0 - weight_ir
        target_intensity = weight_ir * img_ir + weight_vis * img_vis
        loss_intensity = F.l1_loss(img_fused, target_intensity)

        grad_fused = self.sobel(img_fused)
        grad_ir = self.sobel(img_ir)
        grad_vis = self.sobel(img_vis)
        target_grad = weight_ir * grad_ir + weight_vis * grad_vis
        source_grad = target_grad.detach()
        source_grad = source_grad / (source_grad.amax(dim=(-2, -1), keepdim=True) + 1e-6)
        gradient_weight_map = 1.0 + 2.0 * source_grad
        loss_gradient = (gradient_weight_map * (grad_fused - target_grad).abs()).mean()

        lap_fused = self.laplacian(img_fused)
        lap_ir = self.laplacian(img_ir)
        lap_vis = self.laplacian(img_vis)
        target_lap = weight_ir * lap_ir + weight_vis * lap_vis
        edge_weight_map = torch.ones_like(target_grad)
        if objectness_map is not None:
            objectness_map = F.interpolate(objectness_map, size=img_fused.shape[-2:], mode="bilinear", align_corners=False)
            edge_weight_map = edge_weight_map + 0.25 * objectness_map.detach().clamp(0.0, 1.0)
        loss_edge = (
            edge_weight_map * ((grad_fused - target_grad).abs() + (lap_fused - target_lap).abs())
        ).mean()

        loss_ssim = 0.5 * (self.ssim(img_fused, img_ir) + self.ssim(img_fused, img_vis))

        loss_objectness = img_fused.new_zeros(())
        if objectness_map is not None:
            target_obj = F.interpolate(target_grad, size=objectness_map.shape[-2:], mode="bilinear", align_corners=False)
            target_obj = target_obj / (target_obj.amax(dim=(-2, -1), keepdim=True) + 1e-6)
            loss_objectness = F.l1_loss(objectness_map, target_obj.detach())

        loss_detail = img_fused.new_zeros(())
        if detail_map is not None:
            detail_map = F.interpolate(detail_map, size=img_fused.shape[-2:], mode="bilinear", align_corners=False)
            detail_map = detail_map.clamp(0.0, 1.0)
            if objectness_map is None:
                detail_weight_map = torch.ones_like(img_fused)
            else:
                detail_weight_map = objectness_map.detach().clamp(0.0, 1.0)
            fused_detail = torch.abs(img_fused - F.avg_pool2d(img_fused, kernel_size=5, stride=1, padding=2))
            loss_detail = (detail_weight_map * (fused_detail - detail_map).abs()).mean()

        loss_semantic = self._semantic_consistency_loss(img_fused, img_ir, img_vis, semantic_detector)
        total = (
            loss_intensity
            + self.gradient_weight * loss_gradient
            + self.edge_weight * loss_edge
            + self.ssim_weight * loss_ssim
            + self.objectness_weight * loss_objectness
            + self.detail_weight * loss_detail
            + self.semantic_weight * loss_semantic
        )
        components = {
            "intensity": loss_intensity,
            "gradient": loss_gradient,
            "edge": loss_edge,
            "ssim": loss_ssim,
            "objectness": loss_objectness,
            "detail": loss_detail,
            "semantic": loss_semantic,
        }
        return total, components
