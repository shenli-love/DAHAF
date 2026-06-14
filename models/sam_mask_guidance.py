from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class SAMMaskGuidanceBranch(nn.Module):
    """Pixel-level SAM-style mask guidance branch.

    The branch accepts an infrared image and an optional weak pixel prompt
    (for example an offline SAM mask). When no prompt is available it predicts
    a soft spatial prior from the infrared stream, keeping the fusion pipeline
    pixel-driven instead of box-driven.
    """

    def __init__(self, in_channels=1, hidden_channels=32, prompt_channels=1):
        super().__init__()
        self.prompt_channels = prompt_channels
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels + prompt_channels, hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=2, dilation=2, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.SiLU(inplace=True),
        )
        self.mask_head = nn.Conv2d(hidden_channels, 1, kernel_size=1)
        self.refine = nn.Sequential(
            nn.Conv2d(hidden_channels + 1, hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_channels, 1, kernel_size=3, padding=1),
        )

    @staticmethod
    def _boundary(mask):
        grad_x = mask[:, :, :, 1:] - mask[:, :, :, :-1]
        grad_x = F.pad(grad_x.abs(), (0, 1, 0, 0))
        grad_y = mask[:, :, 1:, :] - mask[:, :, :-1, :]
        grad_y = F.pad(grad_y.abs(), (0, 0, 0, 1))
        return (grad_x + grad_y).clamp(0.0, 1.0)

    def _prepare_prompt(self, img_ir, prompt_mask):
        if prompt_mask is None:
            return img_ir.new_zeros(img_ir.shape[0], self.prompt_channels, img_ir.shape[2], img_ir.shape[3])
        if prompt_mask.dim() == 3:
            prompt_mask = prompt_mask.unsqueeze(1)
        prompt_mask = prompt_mask.to(device=img_ir.device, dtype=img_ir.dtype)
        prompt_mask = F.interpolate(prompt_mask, size=img_ir.shape[-2:], mode="bilinear", align_corners=False)
        return prompt_mask.clamp(0.0, 1.0)

    def forward(self, img_ir, prompt_mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        prompt = self._prepare_prompt(img_ir, prompt_mask)
        feat = self.encoder(torch.cat([img_ir, prompt], dim=1))
        coarse_logits = self.mask_head(feat)
        coarse_mask = torch.sigmoid(coarse_logits)
        refined_logits = self.refine(torch.cat([feat, coarse_mask], dim=1))
        mask = torch.sigmoid(coarse_logits + refined_logits)
        return {
            "mask": mask,
            "boundary": self._boundary(mask),
            "logits": coarse_logits + refined_logits,
            "prompt": prompt,
        }
