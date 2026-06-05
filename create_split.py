import os
import glob

# 数据集路径
ir_dir = os.path.join("datasets", "M3FD_Detection", "ir")
meta_dir = os.path.join("datasets", "M3FD_Detection", "meta")

# 获取所有红外图像的名称(不带扩展名)
image_files = glob.glob(os.path.join(ir_dir, "*.png"))
print(f"找到 {len(image_files)} 张图像")

# 提取文件名(不含扩展名)并排序
names = sorted([os.path.splitext(os.path.basename(f))[0] for f in image_files])
print(f"处理 {len(names)} 个样本")

# 按9:1比例分割训练集和验证集
split_idx = int(0.9 * len(names))
train_names = names[:split_idx]
val_names = names[split_idx:]

print(f"训练集: {len(train_names)} 张")
print(f"验证集: {len(val_names)} 张")

# 创建meta目录(如果不存在)
os.makedirs(meta_dir, exist_ok=True)

# 写入train.txt
with open(os.path.join(meta_dir, "train.txt"), "w", encoding="utf-8") as f:
    for name in train_names:
        f.write(name + "\n")

# 写入val.txt
with open(os.path.join(meta_dir, "val.txt"), "w", encoding="utf-8") as f:
    for name in val_names:
        f.write(name + "\n")

print("✓ 已创建 train.txt 和 val.txt")
print(f"  - train.txt: {os.path.join(meta_dir, 'train.txt')}")
print(f"  - val.txt: {os.path.join(meta_dir, 'val.txt')}")