import argparse
import os
import time
import sys

# 获取当前脚本所在目录的上一级（即项目根目录 DAHAF-Net）
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)

# 将项目根目录添加到系统路径
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 设置 PyTorch 显存分配器以减少碎片（参考 TarDAL 最佳实践）
os.environ['PYTORCH_ALLOC_CONF'] = 'expandable_segments:True'


import torch
import yaml
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from models.dahaf_net import DAHAFNet
from utils.dataset import DEFAULT_CLASSES, M3FDDataset, collate_fn
from utils.loss import FusionLoss


def parse_args():
    parser = argparse.ArgumentParser(description="Train DAHAF-Net")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--output", type=str, default="checkpoints")
    return parser.parse_args()


def load_config(path):
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def build_model(cfg):
    return DAHAFNet(
        inp_channels=cfg["MODEL"]["in_channels"],
        dims=tuple(cfg["MODEL"]["dims"]),
        num_classes=cfg["DETECTION"]["num_classes"],
        detector_cfg=cfg["DETECTION"]["detector_cfg"],
        detector_input_size=cfg["DETECTION"]["detector_input_size"],
        detector_box_gain=cfg["DETECTION"]["loss_box"],
        detector_cls_gain=cfg["DETECTION"]["loss_cls"],
        detector_dfl_gain=cfg["DETECTION"]["loss_dfl"],
        detector_pretrained=cfg["DETECTION"]["detector_pretrained"],
        detector_freeze=cfg["DETECTION"]["freeze"],
        detector_initial_no_grad=cfg["DETECTION"]["initial_no_grad"],
        task_dim=cfg["DETECTION"]["task_dim"],
        bridge_channels=cfg["DETECTION"]["bridge_channels"],
        objectness_guidance_alpha=cfg["DETECTION"].get("objectness_guidance_alpha", 0.05),
    )


def build_loader(cfg, split):
    dataset = M3FDDataset(
        root=cfg["DATA"]["root"],
        split=split,
        image_size=tuple(cfg["DATA"]["image_size"]),
        classes=DEFAULT_CLASSES,
    )
    return DataLoader(
        dataset,
        batch_size=cfg["TRAIN"]["batch_size"],
        shuffle=(split == "train"),
        num_workers=cfg["DATA"]["num_workers"],
        pin_memory=cfg["DATA"]["pin_memory"],
        collate_fn=collate_fn,
        drop_last=(split == "train"),
    )


def save_checkpoint(path, epoch, model, optimizer, scheduler, best_val):
    torch.save(
        {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "best_val_loss": best_val,
        },
        path,
    )


def load_compatible_state_dict(model, state_dict):
    model_state = model.state_dict()
    compatible = {
        key: value
        for key, value in state_dict.items()
        if key in model_state and model_state[key].shape == value.shape
    }
    return model.load_state_dict(compatible, strict=False)


def get_stage_weights(cfg, epoch):
    train_cfg = cfg["TRAIN"]
    warmup_epochs = train_cfg.get("warmup_epochs", 10)
    det_start_epoch = train_cfg.get("det_start_epoch", 40)
    det_full_epoch = train_cfg.get("det_full_epoch", 60)

    fusion_scale = min(1.0, epoch / max(warmup_epochs, 1))

    if epoch < det_start_epoch:
        det_scale = 0.0
    else:
        det_scale = min(1.0, (epoch - det_start_epoch + 1) / max(det_full_epoch - det_start_epoch + 1, 1))

    return {
        "fusion": train_cfg["lambda_fusion"] * fusion_scale,
        "det": train_cfg["lambda_det"] * det_scale,
    }


def summarize_fusion_stats(fusion_stats):
    if fusion_stats is None:
        return {}

    summary = {}
    for level_name, level_stats in fusion_stats.items():
        w_ir = level_stats["w_ir"].detach()
        w_vis = level_stats["w_vis"].detach()
        router = level_stats["router"].detach()
        summary[f"{level_name}_mce_ir"] = w_ir.mean().item()
        summary[f"{level_name}_mce_vis"] = w_vis.mean().item()
        summary[f"{level_name}_router_std"] = router.std(dim=(-2, -1)).mean().item()
        summary[f"{level_name}_router_bias"] = (router[:, 0] - router[:, 1]).abs().mean().item()
        if router.shape[1] >= 4:
            summary[f"{level_name}_detail_bias"] = (router[:, 2] - router[:, 3]).abs().mean().item()
    return summary


def train_one_epoch(model, loader, optimizer, fusion_loss, device, cfg, epoch):
    model.train()
    totals = {"loss": 0.0, "fusion": 0.0, "det": 0.0}
    stage_weights = get_stage_weights(cfg, epoch)
    for step, (img_ir, img_vis, targets) in enumerate(loader, start=1):
        img_ir = img_ir.to(device, non_blocking=True)
        img_vis = img_vis.to(device, non_blocking=True)

        outputs = model(img_ir, img_vis, targets=targets, run_detection=True)
        fused = outputs["fused"]
        objectness_map = None
        if outputs["task_signals"] is not None:
            objectness_map = outputs["task_signals"].get("obj")
        loss_fusion, fusion_components = fusion_loss(fused, img_ir, img_vis, objectness_map=objectness_map)
        loss_det, det_metrics = model.detection_loss(outputs, targets)

        total_loss = (
            stage_weights["fusion"] * loss_fusion +
            stage_weights["det"] * loss_det
        )

        optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["TRAIN"]["grad_clip"])
        optimizer.step()

        totals["loss"] += total_loss.item()
        totals["fusion"] += loss_fusion.item()
        totals["det"] += loss_det.item()

        if step % cfg["LOG"]["print_interval"] == 0:
            fusion_stats = summarize_fusion_stats(outputs.get("fusion_stats"))
            print(
                f"Epoch[{epoch}] Step[{step}/{len(loader)}] "
                f"loss={total_loss.item():.4f} fusion={loss_fusion.item():.4f} "
                f"det={loss_det.item():.4f} "
                f"int={fusion_components['intensity'].item():.4f} "
                f"grad={fusion_components['gradient'].item():.4f} "
                f"edge={fusion_components['edge'].item():.4f} "
                f"ssim={fusion_components['ssim'].item():.4f} "
                f"obj={fusion_components['objectness'].item():.4f} "
                f"box={det_metrics['box'].item():.4f} cls={det_metrics['cls'].item():.4f} "
                f"dfl={det_metrics['dfl'].item():.4f} "
                f"s_mce=({fusion_stats.get('shallow_mce_ir', 0.0):.3f},{fusion_stats.get('shallow_mce_vis', 0.0):.3f}) "
                f"d_mce=({fusion_stats.get('deep_mce_ir', 0.0):.3f},{fusion_stats.get('deep_mce_vis', 0.0):.3f}) "
                f"s_bias={fusion_stats.get('shallow_router_bias', 0.0):.3f} "
                f"d_bias={fusion_stats.get('deep_router_bias', 0.0):.3f} "
                f"s_dtl={fusion_stats.get('shallow_detail_bias', 0.0):.3f} "
                f"d_dtl={fusion_stats.get('deep_detail_bias', 0.0):.3f}"
            )

    for key in totals:
        totals[key] /= max(len(loader), 1)
    return totals


@torch.no_grad()
def validate(model, loader, fusion_loss, device, cfg):
    model.eval()
    totals = {"loss": 0.0, "fusion": 0.0, "det": 0.0}
    stage_weights = get_stage_weights(cfg, cfg["TRAIN"]["epochs"])
    for img_ir, img_vis, targets in loader:
        img_ir = img_ir.to(device, non_blocking=True)
        img_vis = img_vis.to(device, non_blocking=True)
        outputs = model(img_ir, img_vis, targets=targets, run_detection=True)
        fused = outputs["fused"]
        objectness_map = None
        if outputs["task_signals"] is not None:
            objectness_map = outputs["task_signals"].get("obj")
        loss_fusion, _ = fusion_loss(fused, img_ir, img_vis, objectness_map=objectness_map)
        loss_det, _ = model.detection_loss(outputs, targets)
        total_loss = (
            stage_weights["fusion"] * loss_fusion +
            stage_weights["det"] * loss_det
        )
        totals["loss"] += total_loss.item()
        totals["fusion"] += loss_fusion.item()
        totals["det"] += loss_det.item()

    for key in totals:
        totals[key] /= max(len(loader), 1)
    return totals


def main():
    args = parse_args()
    cfg = load_config(args.config)
    device = torch.device(args.device)

    # 清理 CUDA 缓存（参考 TarDAL 最佳实践）
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    os.makedirs(args.output, exist_ok=True)
    train_loader = build_loader(cfg, "train")
    val_loader = build_loader(cfg, "val")

    model = build_model(cfg).to(device)
    optimizer = AdamW(model.parameters(), lr=cfg["TRAIN"]["learning_rate"], weight_decay=cfg["TRAIN"]["weight_decay"])
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg["TRAIN"]["epochs"], eta_min=cfg["TRAIN"]["min_lr"])
    fusion_loss = FusionLoss(
        gradient_weight=cfg["TRAIN"]["gradient_weight"],
        edge_weight=cfg["TRAIN"].get("edge_weight", 2.0),
        ssim_weight=cfg["TRAIN"].get("ssim_weight", 1.0),
        objectness_weight=cfg["TRAIN"].get("objectness_weight", 0.2),
    ).to(device)

    start_epoch = 1
    best_val = float("inf")
    if args.resume and os.path.isfile(args.resume):
        checkpoint = torch.load(args.resume, map_location=device)
        load_compatible_state_dict(model, checkpoint["model"])
        try:
            optimizer.load_state_dict(checkpoint["optimizer"])
        except Exception as exc:
            print(f"[resume] skip optimizer state: {exc}")
        try:
            scheduler.load_state_dict(checkpoint["scheduler"])
        except Exception as exc:
            print(f"[resume] skip scheduler state: {exc}")
        start_epoch = checkpoint.get("epoch", 0) + 1
        best_val = checkpoint.get("best_val_loss", float("inf"))

    writer = SummaryWriter(log_dir=os.path.join(args.output, "logs"))

    for epoch in range(start_epoch, cfg["TRAIN"]["epochs"] + 1):
        tic = time.time()
        train_stats = train_one_epoch(model, train_loader, optimizer, fusion_loss, device, cfg, epoch)
        scheduler.step()
        val_stats = validate(model, val_loader, fusion_loss, device, cfg)

        writer.add_scalar("train/total", train_stats["loss"], epoch)
        writer.add_scalar("train/fusion", train_stats["fusion"], epoch)
        writer.add_scalar("train/det", train_stats["det"], epoch)
        writer.add_scalar("val/total", val_stats["loss"], epoch)
        writer.add_scalar("val/fusion", val_stats["fusion"], epoch)
        writer.add_scalar("val/det", val_stats["det"], epoch)

        latest_path = os.path.join(args.output, "latest.pth")
        save_checkpoint(latest_path, epoch, model, optimizer, scheduler, best_val)
        if epoch % cfg["LOG"]["save_interval"] == 0:
            save_checkpoint(os.path.join(args.output, f"epoch_{epoch}.pth"), epoch, model, optimizer, scheduler, best_val)
        if val_stats["loss"] < best_val:
            best_val = val_stats["loss"]
            save_checkpoint(os.path.join(args.output, "best.pth"), epoch, model, optimizer, scheduler, best_val)

        print(
            f"Epoch {epoch} | "
            f"train={train_stats['loss']:.4f} val={val_stats['loss']:.4f} | "
            f"time={time.time() - tic:.1f}s"
        )

    writer.close()


if __name__ == "__main__":
    main()
