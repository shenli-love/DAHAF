import torch
import torch.nn as nn


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


class LowFreqExtractor(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block1 = C2f(channels, num_blocks=6)
        self.block2 = C2f(channels, num_blocks=6)

    def forward(self, x):
        return self.block2(self.block1(x))


class AffineCoupling(nn.Module):
    def __init__(self, channels):
        super().__init__()
        half = channels // 2
        self.shuffle = nn.Conv2d(channels, channels, kernel_size=1, bias=True)
        self.phi = self._make_subnet(half)
        self.rho = self._make_subnet(half)
        self.eta = self._make_subnet(half)

    @staticmethod
    def _make_subnet(channels):
        return nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            make_norm(channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False),
            make_norm(channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
        )

    def forward(self, x):
        z1, z2 = self.shuffle(x).chunk(2, dim=1)
        z2 = z2 + self.phi(z1)
        scale = torch.tanh(self.rho(z2))
        z1 = z1 * torch.exp(scale) + self.eta(z2)
        return torch.cat([z1, z2], dim=1)


class HighFreqExtractor(nn.Module):
    def __init__(self, channels, num_layers=3):
        super().__init__()
        self.layers = nn.Sequential(*[AffineCoupling(channels) for _ in range(num_layers)])

    def forward(self, x):
        return self.layers(x)


class EncoderStage(nn.Module):
    def __init__(self, in_channels, out_channels, num_extractors=1, downsample=False):
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
            layers.append(LowFreqExtractor(out_channels))
        self.body = nn.Sequential(*layers)
        self.base_branch = LowFreqExtractor(out_channels)
        self.detail_branch = HighFreqExtractor(out_channels)

    def forward(self, x):
        feat = self.body(x)
        base = self.base_branch(feat)
        detail = self.detail_branch(feat)
        return feat, base, detail


class DE_Encoder(nn.Module):
    def __init__(self, inp_channels=1, dims=(64, 128, 256)):
        super().__init__()
        self.patch_embed = OverlapPatchEmbed(inp_channels, dims[0])
        self.stage1 = EncoderStage(dims[0], dims[0], num_extractors=5, downsample=False)
        self.stage2 = EncoderStage(dims[0], dims[1], num_extractors=2, downsample=True)
        self.stage3 = EncoderStage(dims[1], dims[2], num_extractors=1, downsample=True)

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
