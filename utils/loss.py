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
        objectness_weight=0.2,
    ):
        super().__init__()
        self.sobel = SobelXY()
        self.ssim = SSIMLoss()
        self.gradient_weight = gradient_weight
        self.edge_weight = edge_weight
        self.ssim_weight = ssim_weight
        self.objectness_weight = objectness_weight

    def forward(self, img_fused, img_ir, img_vis, objectness_map=None):
        target_intensity = torch.maximum(img_ir, img_vis)
        loss_intensity = F.l1_loss(img_fused, target_intensity)

        grad_fused = self.sobel(img_fused)
        grad_ir = self.sobel(img_ir)
        grad_vis = self.sobel(img_vis)
        target_grad = torch.maximum(grad_ir, grad_vis)
        loss_gradient = F.l1_loss(grad_fused, target_grad)

        edge_weight_map = torch.ones_like(target_grad)
        if objectness_map is not None:
            objectness_map = F.interpolate(objectness_map, size=img_fused.shape[-2:], mode="bilinear", align_corners=False)
            edge_weight_map = edge_weight_map + 0.25 * objectness_map.detach().clamp(0.0, 1.0)
        loss_edge = (edge_weight_map * (grad_fused - target_grad).abs()).mean()

        loss_ssim = 0.5 * (self.ssim(img_fused, img_ir) + self.ssim(img_fused, img_vis))

        loss_objectness = img_fused.new_zeros(())
        if objectness_map is not None:
            target_obj = F.interpolate(target_grad, size=objectness_map.shape[-2:], mode="bilinear", align_corners=False)
            target_obj = target_obj / (target_obj.amax(dim=(-2, -1), keepdim=True) + 1e-6)
            loss_objectness = F.l1_loss(objectness_map, target_obj.detach())

        total = (
            loss_intensity
            + self.gradient_weight * loss_gradient
            + self.edge_weight * loss_edge
            + self.ssim_weight * loss_ssim
            + self.objectness_weight * loss_objectness
        )
        components = {
            "intensity": loss_intensity,
            "gradient": loss_gradient,
            "edge": loss_edge,
            "ssim": loss_ssim,
            "objectness": loss_objectness,
        }
        return total, components
