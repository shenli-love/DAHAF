import torch
import torch.nn as nn
import torch.nn.functional as F


def make_norm(channels, num_groups=8):
    groups = min(num_groups, channels)
    while channels % groups != 0 and groups > 1:
        groups -= 1
    return nn.GroupNorm(groups, channels)


class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_channels=1, embed_dim=64):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=3, stride=1, padding=1, bias=False)
        self.norm = make_norm(embed_dim)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.act(self.norm(self.proj(x)))


class Bottleneck(nn.Module):
    def __init__(self, channels, shortcut=True, expansion=0.5, groups=1):
        super().__init__()
        hidden = int(channels * expansion)
        self.cv1 = nn.Conv2d(channels, hidden, kernel_size=1, bias=False)
        self.bn1 = make_norm(hidden)
        self.cv2 = nn.Conv2d(hidden, channels, kernel_size=3, padding=1, groups=groups, bias=False)
        self.bn2 = make_norm(channels)
        self.act = nn.SiLU(inplace=True)
        self.shortcut = shortcut

    def forward(self, x):
        out = self.act(self.bn1(self.cv1(x)))
        out = self.act(self.bn2(self.cv2(out)))
        return x + out if self.shortcut else out


class C2f(nn.Module):
    def __init__(self, channels, num_blocks=6, expansion=0.5):
        super().__init__()
        hidden = int(channels * expansion)
        self.cv1 = nn.Conv2d(channels, hidden * 2, kernel_size=1, bias=False)
        self.cv2 = nn.Conv2d(hidden * (2 + num_blocks), channels, kernel_size=1, bias=False)
        self.bn = make_norm(channels)
        self.act = nn.SiLU(inplace=True)
        self.blocks = nn.ModuleList([Bottleneck(hidden, shortcut=True, expansion=1.0) for _ in range(num_blocks)])

    def forward(self, x):
        parts = list(self.cv1(x).chunk(2, dim=1))
        for block in self.blocks:
            parts.append(block(parts[-1]))
        return self.act(self.bn(self.cv2(torch.cat(parts, dim=1))))


class FeatureExtractor(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block1 = C2f(channels, num_blocks=6)
        self.block2 = C2f(channels, num_blocks=6)

    def forward(self, x):
        return self.block2(self.block1(x))


class GuidedFilter(nn.Module):
    def __init__(self, radius, eps):
        super().__init__()
        self.radius = int(radius)
        self.eps = float(eps)

    def _box_filter(self, x):
        if self.radius <= 0:
            return x
        radius_h = min(self.radius, max((x.shape[-2] - 1) // 2, 0))
        radius_w = min(self.radius, max((x.shape[-1] - 1) // 2, 0))
        if radius_h == 0 and radius_w == 0:
            return x
        kernel_size = (radius_h * 2 + 1, radius_w * 2 + 1)
        padding = (radius_h, radius_w)
        return F.avg_pool2d(x, kernel_size=kernel_size, stride=1, padding=padding, count_include_pad=False)

    def forward(self, guide, src):
        mean_i = self._box_filter(guide)
        mean_p = self._box_filter(src)
        mean_ip = self._box_filter(guide * src)
        cov_ip = mean_ip - mean_i * mean_p

        mean_ii = self._box_filter(guide * guide)
        var_i = mean_ii - mean_i * mean_i

        a = cov_ip / (var_i + self.eps)
        b = mean_p - a * mean_i
        mean_a = self._box_filter(a)
        mean_b = self._box_filter(b)

        low = mean_a * guide + mean_b
        high = src - low
        return low, high


class GuidedFrequencyDecomposition(nn.Module):
    def __init__(self, channels, radius, eps, detail_gain=1.0):
        super().__init__()
        self.guided = GuidedFilter(radius=radius, eps=eps)
        self.low_proj = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False),
            make_norm(channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            make_norm(channels),
            nn.SiLU(inplace=True),
        )
        self.high_proj = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False),
            make_norm(channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            make_norm(channels),
            nn.SiLU(inplace=True),
        )
        self.mix = nn.Sequential(
            nn.Conv2d(channels * 3, channels, kernel_size=1, bias=False),
            make_norm(channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            make_norm(channels),
            nn.SiLU(inplace=True),
        )
        self.detail_scale = nn.Parameter(torch.tensor(float(detail_gain)))

    def forward(self, x):
        low, high = self.guided(x, x)
        low = self.low_proj(low)
        high = self.high_proj(high)
        detail = (0.5 + torch.sigmoid(self.detail_scale)) * high
        fused = x + self.mix(torch.cat([x, low, detail], dim=1))
        return fused, low, detail


class EncoderStage(nn.Module):
    def __init__(self, in_channels, out_channels, num_extractors=1, downsample=False, radius=3, eps=1e-3, detail_gain=1.0):
        super().__init__()
        layers = []
        if downsample:
            layers.extend([
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False),
                make_norm(out_channels),
                nn.SiLU(inplace=True),
            ])
        elif in_channels != out_channels:
            layers.extend([
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                make_norm(out_channels),
                nn.SiLU(inplace=True),
            ])

        for _ in range(num_extractors):
            layers.append(FeatureExtractor(out_channels))
        self.body = nn.Sequential(*layers)
        self.decomposition = GuidedFrequencyDecomposition(
            out_channels,
            radius=radius,
            eps=eps,
            detail_gain=detail_gain,
        )

    def forward(self, x):
        feat = self.body(x)
        fused, base, detail = self.decomposition(feat)
        return fused, base, detail


class DE_Encoder(nn.Module):
    def __init__(self, inp_channels=1, dims=(64, 128, 256)):
        super().__init__()
        self.patch_embed = OverlapPatchEmbed(inp_channels, dims[0])
        self.stage1 = EncoderStage(dims[0], dims[0], num_extractors=5, downsample=False, radius=7, eps=5e-3, detail_gain=1.2)
        self.stage2 = EncoderStage(dims[0], dims[1], num_extractors=2, downsample=True, radius=5, eps=2e-3, detail_gain=1.0)
        self.stage3 = EncoderStage(dims[1], dims[2], num_extractors=1, downsample=True, radius=3, eps=1e-3, detail_gain=0.8)

    def forward(self, x):
        x0 = self.patch_embed(x)
        stage1, base1, detail1 = self.stage1(x0)
        stage2, base2, detail2 = self.stage2(stage1)
        stage3, base3, detail3 = self.stage3(stage2)
        return {
            "stage1": stage1,
            "stage2": stage2,
            "stage3": stage3,
            "base": [base1, base2, base3],
            "detail": [detail1, detail2, detail3],
            "mid": stage2,
        }
