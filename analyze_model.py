"""模型资源分析工具 - 计算参数量、FLOPs和显存占用"""
import os
import sys
import torch
import yaml

# 获取项目根目录
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from models.dahaf_net import DAHAFNet


def load_config(path="configs/default.yaml"):
    """加载配置文件"""
    config_path = os.path.join(current_dir, path)
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def count_parameters(model):
    """统计模型参数量"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_params = total_params - trainable_params
    return total_params, trainable_params, non_trainable_params


def calculate_flops(model, device):
    """计算FLOPs（使用thop库）"""
    try:
        from thop import profile, clever_format
        
        dummy_ir = torch.randn(1, 1, 256, 256).to(device)
        dummy_vis = torch.randn(1, 1, 256, 256).to(device)
        
        # 关闭检测以减少计算量，只计算融合网络
        flops, params = profile(model, inputs=(dummy_ir, dummy_vis), verbose=False)
        flops_fmt, params_fmt = clever_format([flops, params])
        
        return flops, params, flops_fmt, params_fmt
    except ImportError:
        print("⚠️  警告: thop 未安装，跳过 FLOPs 计算")
        print("   安装命令: pip install thop")
        return None, None, None, None


def test_memory_usage(model, device, batch_sizes=[1, 2, 4, 8]):
    """测试不同batch size下的显存占用"""
    if not torch.cuda.is_available():
        print("⚠️  CUDA 不可用，跳过显存测试")
        return
    
    print("\n" + "="*70)
    print("🔍 不同 Batch Size 资源需求测试")
    print("="*70)
    print(f"{'Batch Size':<12} {'Peak Memory (MB)':<18} {'Inference Time (ms)':<20}")
    print("-"*70)
    
    results = []
    for bs in batch_sizes:
        # 清理缓存
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        batch_ir = torch.randn(bs, 1, 256, 256).to(device)
        batch_vis = torch.randn(bs, 1, 256, 256).to(device)
        
        # 预热
        with torch.no_grad():
            _ = model(batch_ir, batch_vis, run_detection=False)
        
        # 正式测试
        torch.cuda.synchronize()
        import time
        start_time = time.time()
        
        with torch.no_grad():
            _ = model(batch_ir, batch_vis, run_detection=False)
        
        torch.cuda.synchronize()
        elapsed_ms = (time.time() - start_time) * 1000
        
        peak_mem_mb = torch.cuda.max_memory_allocated() / (1024**2)
        
        print(f"{bs:<12} {peak_mem_mb:<18.2f} {elapsed_ms:<20.2f}")
        results.append({
            'batch_size': bs,
            'peak_memory_mb': peak_mem_mb,
            'inference_time_ms': elapsed_ms
        })
    
    print("="*70)
    return results


def analyze_model():
    """主分析函数"""
    print("\n" + "="*70)
    print("📊 DAHAF-Net 模型资源分析报告")
    print("="*70)
    
    # 加载配置
    try:
        cfg = load_config()
        print(f"✅ 配置文件加载成功: configs/default.yaml")
    except Exception as e:
        print(f"❌ 配置文件加载失败: {e}")
        print("   使用默认配置...")
        cfg = {
            "MODEL": {
                "in_channels": 1,
                "dims": [64, 128, 256],
                "contrastive_dim": 128,
                "contrastive_proj": 64,
            },
            "DETECTION": {
                "num_classes": 6,
                "detector_cfg": "yolo11n.yaml",
                "detector_input_size": 256,
                "detector_pretrained": "yolo11n.pt",
                "freeze": True,
                "task_dim": 64,
                "bridge_channels": 128,
            }
        }
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  运行设备: {device}")
    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
    
    # 构建模型
    print("\n🔧 正在构建模型...")
    model = DAHAFNet(
        inp_channels=cfg["MODEL"]["in_channels"],
        dims=tuple(cfg["MODEL"]["dims"]),
        num_classes=cfg["DETECTION"]["num_classes"],
        detector_cfg=cfg["DETECTION"]["detector_cfg"],
        detector_input_size=cfg["DETECTION"]["detector_input_size"],
        detector_pretrained=cfg["DETECTION"]["detector_pretrained"],
        detector_freeze=cfg["DETECTION"]["freeze"],
        task_dim=cfg["DETECTION"]["task_dim"],
        bridge_channels=cfg["DETECTION"]["bridge_channels"],
        objectness_guidance_alpha=cfg["DETECTION"].get("objectness_guidance_alpha", 0.05),
    ).to(device)
    
    model.eval()
    print("✅ 模型构建完成")
    
    # 1. 参数量统计
    print("\n" + "="*70)
    print("📏 参数量统计")
    print("="*70)
    total_params, trainable_params, non_trainable_params = count_parameters(model)
    
    print(f"总参数量:       {total_params:>15,}  ({total_params/1e6:.2f} M)")
    print(f"可训练参数:     {trainable_params:>15,}  ({trainable_params/1e6:.2f} M)")
    print(f"冻结参数:       {non_trainable_params:>15,}  ({non_trainable_params/1e6:.2f} M)")
    
    if total_params > 0:
        trainable_ratio = trainable_params / total_params * 100
        print(f"可训练比例:     {trainable_ratio:>14.2f}%")
    
    # 按模块统计
    print("\n📋 各模块参数量分布:")
    print("-"*70)
    module_params = {}
    for name, module in model.named_children():
        params = sum(p.numel() for p in module.parameters())
        module_params[name] = params
        percentage = params / total_params * 100 if total_params > 0 else 0
        print(f"  {name:<30s} {params:>12,}  ({percentage:>5.2f}%)")
    
    # 2. FLOPs 计算
    print("\n" + "="*70)
    print("⚡ FLOPs 计算")
    print("="*70)
    flops, params_thop, flops_fmt, params_fmt = calculate_flops(model, device)
    
    if flops is not None:
        print(f"FLOPs:          {flops_fmt:>20s}  ({flops/1e9:.2f} G)")
        print(f"Params (thop):  {params_fmt:>20s}  ({params_thop/1e6:.2f} M)")
    
    # 3. 显存测试
    memory_results = test_memory_usage(model, device, batch_sizes=[1, 2, 4, 8])
    
    # 4. 总结
    print("\n" + "="*70)
    print("📝 总结")
    print("="*70)
    print(f"模型名称:       DAHAF-Net")
    print(f"输入尺寸:       IR: (1, 1, 256, 256), VIS: (1, 1, 256, 256)")
    print(f"总参数量:       {total_params/1e6:.2f} M")
    if flops is not None:
        print(f"FLOPs:          {flops/1e9:.2f} G")
    if memory_results and len(memory_results) > 0:
        print(f"Batch=1 显存:   {memory_results[0]['peak_memory_mb']:.2f} MB")
        print(f"Batch=4 显存:   {memory_results[min(3, len(memory_results)-1)]['peak_memory_mb']:.2f} MB")
    print("="*70 + "\n")


if __name__ == "__main__":
    analyze_model()
