import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.proj = None
        if in_channels != out_channels:
            self.proj = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.act = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x):
        identity = x if self.proj is None else self.proj(x)
        out = self.block(x)
        return self.act(out + identity)


class UpBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.fuse = ResidualConvBlock(in_channels + skip_channels, out_channels)

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[-2:], mode="nearest")
        return self.fuse(torch.cat([x, skip], dim=1))


class EdgeRefinementHead(nn.Module):
    def __init__(self, channels, out_channels):
        super().__init__()
        self.refine = nn.Sequential(
            nn.Conv2d(channels + 1, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(channels, out_channels, kernel_size=3, padding=1, bias=False),
        )

    def forward(self, feat, edge_hint):
        edge_hint = F.interpolate(edge_hint, size=feat.shape[-2:], mode="nearest")
        return self.refine(torch.cat([feat, edge_hint], dim=1))


class DE_Decoder(nn.Module):
    def __init__(self, deep_dim=256, mid_dim=128, shallow_dim=64, out_channels=3):
        super().__init__()
        self.stem = ResidualConvBlock(deep_dim, deep_dim)
        self.up_mid = UpBlock(deep_dim, mid_dim, mid_dim)
        self.up_shallow = UpBlock(mid_dim, shallow_dim, shallow_dim)
        self.output = nn.Sequential(
            nn.Conv2d(shallow_dim, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(32, out_channels, kernel_size=3, padding=1),
        )
        self.detail_enhance = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.edge_head = EdgeRefinementHead(shallow_dim, out_channels)
        self.residual_detail = nn.Sequential(
            nn.Conv2d(out_channels + 1, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
        )

    def forward(self, fused_deep, skip_mid, skip_shallow, residual=None):
        feat = self.stem(fused_deep)
        feat = self.up_mid(feat, skip_mid)
        feat = self.up_shallow(feat, skip_shallow)
        out = self.output(feat)

        if residual is not None:
            residual_detail = residual - F.avg_pool2d(residual, kernel_size=3, stride=1, padding=1)
            edge_hint = torch.abs(residual_detail)
            edge_residual = self.edge_head(feat, edge_hint)
            detail_residual = self.residual_detail(torch.cat([out, residual_detail], dim=1))
            out = out + 0.20 * detail_residual + 0.12 * edge_residual
            detail = self.detail_enhance(out)
            out = out + 0.10 * detail

        return out.clamp(0.0, 1.0)
