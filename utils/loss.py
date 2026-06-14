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
        mask_region_weight=0.4,
        mask_boundary_weight=0.8,
        detail_weight=1.0,
        local_window_size=7,
    ):
        super().__init__()
        self.sobel = SobelXY()
        self.laplacian = Laplacian()
        self.ssim = SSIMLoss()
        self.gradient_weight = gradient_weight
        self.edge_weight = edge_weight
        self.ssim_weight = ssim_weight
        self.mask_region_weight = mask_region_weight
        self.mask_boundary_weight = mask_boundary_weight
        self.detail_weight = detail_weight
        self.local_window_size = local_window_size

    def _local_variance(self, x):
        kernel = self.local_window_size
        mean = F.avg_pool2d(x, kernel, stride=1, padding=kernel // 2)
        mean_sq = F.avg_pool2d(x * x, kernel, stride=1, padding=kernel // 2)
        return (mean_sq - mean.pow(2)).clamp_min(0.0)

    @staticmethod
    def _mask_boundary(mask):
        grad_x = mask[:, :, :, 1:] - mask[:, :, :, :-1]
        grad_x = F.pad(grad_x.abs(), (0, 1, 0, 0))
        grad_y = mask[:, :, 1:, :] - mask[:, :, :-1, :]
        grad_y = F.pad(grad_y.abs(), (0, 0, 0, 1))
        return (grad_x + grad_y).clamp(0.0, 1.0)

    def forward(
        self,
        img_fused,
        img_ir,
        img_vis,
        mask_prior=None,
        detail_map=None,
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
        if mask_prior is not None:
            mask_prior = F.interpolate(mask_prior, size=img_fused.shape[-2:], mode="bilinear", align_corners=False)
            mask_prior = mask_prior.clamp(0.0, 1.0)
            edge_weight_map = edge_weight_map + 0.25 * mask_prior.detach()
        loss_edge = (
            edge_weight_map * ((grad_fused - target_grad).abs() + (lap_fused - target_lap).abs())
        ).mean()

        loss_ssim = 0.5 * (self.ssim(img_fused, img_ir) + self.ssim(img_fused, img_vis))

        loss_mask_region = img_fused.new_zeros(())
        loss_mask_boundary = img_fused.new_zeros(())
        if mask_prior is not None:
            mask_detached = mask_prior.detach()
            mask_norm = mask_detached.mean(dim=(-2, -1), keepdim=True).clamp_min(1e-4)
            region_error = (img_fused - target_intensity).abs()
            loss_mask_region = (mask_detached * region_error).mean() / mask_norm.mean()

            boundary = self._mask_boundary(mask_prior).detach()
            boundary_norm = boundary.mean(dim=(-2, -1), keepdim=True).clamp_min(1e-4)
            boundary_grad_target = torch.maximum(grad_ir, grad_vis).detach()
            loss_mask_boundary = (boundary * (grad_fused - boundary_grad_target).abs()).mean() / boundary_norm.mean()

        loss_detail = img_fused.new_zeros(())
        if detail_map is not None:
            detail_map = F.interpolate(detail_map, size=img_fused.shape[-2:], mode="bilinear", align_corners=False)
            detail_map = detail_map.clamp(0.0, 1.0)
            if mask_prior is None:
                detail_weight_map = torch.ones_like(img_fused)
            else:
                detail_weight_map = mask_prior.detach().clamp(0.0, 1.0)
            fused_detail = torch.abs(img_fused - F.avg_pool2d(img_fused, kernel_size=5, stride=1, padding=2))
            loss_detail = (detail_weight_map * (fused_detail - detail_map).abs()).mean()

        total = (
            loss_intensity
            + self.gradient_weight * loss_gradient
            + self.edge_weight * loss_edge
            + self.ssim_weight * loss_ssim
            + self.mask_region_weight * loss_mask_region
            + self.mask_boundary_weight * loss_mask_boundary
            + self.detail_weight * loss_detail
        )
        components = {
            "intensity": loss_intensity,
            "gradient": loss_gradient,
            "edge": loss_edge,
            "ssim": loss_ssim,
            "mask_region": loss_mask_region,
            "mask_boundary": loss_mask_boundary,
            "detail": loss_detail,
        }
        return total, components
