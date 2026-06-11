import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# ------------------- 基础工具函数 -------------------
def make_norm(channels, num_groups=8):
    groups = min(num_groups, channels)
    while channels % groups != 0 and groups > 1:
        groups -= 1
    return nn.GroupNorm(groups, channels)

# ------------------- 基础卷积块 -------------------
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

# ------------------- 低频提取 -------------------
class LowFreqExtractor(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block1 = C2f(channels, num_blocks=6)
        self.block2 = C2f(channels, num_blocks=6)
    def forward(self, x):
        return self.block2(self.block1(x))

# ------------------- 高频提取 -------------------
class HighFreqExtractor(nn.Module):
    """
    高频特征提取器
    Sobel + Laplacian + 深度卷积 + 残差增强
    """
    def __init__(self, channels):
        super().__init__()
        # 卷积增强
        self.enhance = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False),
            make_norm(channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels, channels, 1, bias=False),
            make_norm(channels),
            nn.SiLU(inplace=True),
        )
        # Sobel X
        sobel_x = torch.tensor([
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1]
        ], dtype=torch.float32)
        # Sobel Y
        sobel_y = torch.tensor([
            [-1, -2, -1],
            [0, 0, 0],
            [1, 2, 1]
        ], dtype=torch.float32)
        # Laplacian
        laplacian = torch.tensor([
            [-1, -1, -1],
            [-1, 8, -1],
            [-1, -1, -1]
        ], dtype=torch.float32)
        self.register_buffer("sobel_x", sobel_x.unsqueeze(0).unsqueeze(0))
        self.register_buffer("sobel_y", sobel_y.unsqueeze(0).unsqueeze(0))
        self.register_buffer("laplacian", laplacian.unsqueeze(0).unsqueeze(0))
    def apply_filter(self, x, kernel):
        c = x.shape[1]
        kernel = kernel.repeat(c, 1, 1, 1)
        return F.conv2d(x, kernel, padding=1, groups=c)
    def forward(self, x):
        feat = self.enhance(x)
        edge_x = self.apply_filter(feat, self.sobel_x)
        edge_y = self.apply_filter(feat, self.sobel_y)
        edge = torch.sqrt(edge_x ** 2 + edge_y ** 2 + 1e-6)
        lap = self.apply_filter(feat, self.laplacian)
        high_freq = edge + lap
        out = feat + high_freq
        return out

# ------------------- 编码器阶段 -------------------
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

# ------------------- DE_Encoder -------------------
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

# ------------------- 可视化函数 -------------------
def visualize_features(model, image_path, output_dir='./'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    img = Image.open(image_path).convert('L')
    w, h = img.size
    new_w = (w // 4) * 4
    new_h = (h // 4) * 4
    if new_w != w or new_h != h:
        img = img.resize((new_w, new_h), Image.BICUBIC)
        print(f"图像尺寸调整为 {new_w}x{new_h} (4的倍数)")

    img_np = np.asarray(img, dtype=np.float32) / 255.0
    x = torch.from_numpy(img_np).unsqueeze(0).unsqueeze(0)
    x = (x - 0.5) / 0.5
    x = x.to(device)

    model.eval()
    with torch.no_grad():
        outputs = model(x)

    base_list = outputs['base']
    detail_list = outputs['detail']
    scales = ['stage1 (1x)', 'stage2 (1/2x)', 'stage3 (1/4x)']

    def feat_to_gray(tensor):
        t = tensor.squeeze(0)
        t_abs = torch.abs(t)
        t_map, _ = t_abs.max(dim=0)
        t_map = torch.pow(t_map, 0.7)
        vmin, vmax = t_map.min(), t_map.max()
        if vmax - vmin > 1e-6:
            t_norm = (t_map - vmin) / (vmax - vmin)
        else:
            t_norm = torch.zeros_like(t_map)
        return (t_norm.cpu().numpy() * 255).astype(np.uint8)

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for i in range(3):
        base_gray = feat_to_gray(base_list[i])
        axes[0, i].imshow(base_gray, cmap='gray')
        axes[0, i].set_title(f'Low Freq - {scales[i]}')
        axes[0, i].axis('off')

        detail_gray = feat_to_gray(detail_list[i])
        axes[1, i].imshow(detail_gray, cmap='gray')
        axes[1, i].set_title(f'High Freq - {scales[i]}')
        axes[1, i].axis('off')

    plt.tight_layout()
    save_path = output_dir + 'low_high_freq_features.png'
    plt.savefig(save_path, dpi=150)
    plt.show()
    print(f"特征图已保存至: {save_path}")

    for i, (base, detail) in enumerate(zip(base_list, detail_list)):
        base_img = feat_to_gray(base)
        detail_img = feat_to_gray(detail)
        Image.fromarray(base_img).save(f'{output_dir}low_freq_stage{i + 1}.png')
        Image.fromarray(detail_img).save(f'{output_dir}high_freq_stage{i + 1}.png')
    print("各尺度特征图已单独保存（原始分辨率）。")

# ------------------- 主函数 -------------------
if __name__ == '__main__':
    model = DE_Encoder(inp_channels=1, dims=(64, 128, 256))
    # 可加载预训练权重
    # model.load_state_dict(torch.load('your_weights.pth', map_location='cpu'))
    visualize_features(model, '00061.png', output_dir='D:/machinelearn/infrared_and_visible_image_fusion/Tar/DAHAF-Net/test-files')
