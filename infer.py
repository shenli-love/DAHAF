import argparse
import os
import sys

import cv2
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from models.dahaf_net import DAHAFNet
from utils.dataset import DEFAULT_CLASSES, M3FDDataset, collate_fn, letterbox_resize, undo_letterbox_image


def parse_args():
    parser = argparse.ArgumentParser(description="DAHAF-Net inference")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output", type=str, default="outputs")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num-samples", type=int, default=50)
    parser.add_argument("--with-det", action="store_true")
    parser.add_argument("--ir-path", type=str, default=None, help="Optional single infrared image path.")
    parser.add_argument("--vis-path", type=str, default=None, help="Optional single visible image path.")
    parser.add_argument("--output-name", type=str, default="fused_result")
    return parser.parse_args()


def load_config(path):
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def build_model(cfg, device):
    model = DAHAFNet(
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
    ).to(device)
    return model


def load_checkpoint(model, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    model_state = model.state_dict()
    compatible = {
        key: value
        for key, value in state_dict.items()
        if key in model_state and model_state[key].shape == value.shape
    }
    model.load_state_dict(compatible, strict=False)
    model.eval()


def load_single_grayscale(path, image_size, device):
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise RuntimeError(f"Failed to load image: {path}")
    image, resize_meta = letterbox_resize(image, image_size, interpolation=cv2.INTER_LINEAR)
    tensor = torch.from_numpy(image.astype(np.float32) / 255.0).unsqueeze(0).unsqueeze(0).to(device)
    return tensor, resize_meta


def save_fused_image(fused, resize_meta, save_path):
    fused = fused.squeeze().cpu().numpy()
    fused = undo_letterbox_image(fused, resize_meta, interpolation=cv2.INTER_LANCZOS4)
    fused = (fused * 255.0).clip(0, 255).astype("uint8")
    cv2.imwrite(save_path, fused)


def run_single_image(model, cfg, device, args):
    image_size = tuple(cfg["DATA"]["image_size"])
    img_ir, resize_meta = load_single_grayscale(args.ir_path, image_size, device)
    img_vis, _ = load_single_grayscale(args.vis_path, image_size, device)

    targets = [{
        "boxes": torch.zeros((0, 4), dtype=torch.float32),
        "labels": torch.zeros((0,), dtype=torch.long),
        "image_id": args.output_name,
        "orig_size": torch.tensor([resize_meta["orig_h"], resize_meta["orig_w"]], dtype=torch.float32),
        "resize_meta": {
            key: torch.tensor(value, dtype=torch.float32)
            for key, value in resize_meta.items()
            if key in {"scale", "pad_left", "pad_top", "new_w", "new_h", "orig_w", "orig_h"}
        },
    }]

    with torch.no_grad():
        outputs = model(img_ir, img_vis, targets=targets, run_detection=args.with_det)

    os.makedirs(args.output, exist_ok=True)
    save_path = os.path.join(args.output, f"{args.output_name}.png")
    save_fused_image(outputs["fused"], resize_meta, save_path)
    print(f"Saved fused image to: {save_path}")


def run_dataset_inference(model, cfg, device, args):
    dataset = M3FDDataset(
        root=cfg["DATA"]["root"],
        split="val",
        image_size=tuple(cfg["DATA"]["image_size"]),
        classes=DEFAULT_CLASSES,
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate_fn)

    os.makedirs(args.output, exist_ok=True)
    with torch.no_grad():
        for idx, (img_ir, img_vis, targets) in enumerate(loader):
            if idx >= args.num_samples:
                break
            img_ir = img_ir.to(device)
            img_vis = img_vis.to(device)
            outputs = model(img_ir, img_vis, targets=targets, run_detection=args.with_det)

            image_id = targets[0]["image_id"]
            save_path = os.path.join(args.output, f"{image_id}_fused.png")
            resize_meta = {
                "scale": float(targets[0]["resize_meta"]["scale"]),
                "pad_left": float(targets[0]["resize_meta"]["pad_left"]),
                "pad_top": float(targets[0]["resize_meta"]["pad_top"]),
                "new_w": float(targets[0]["resize_meta"]["new_w"]),
                "new_h": float(targets[0]["resize_meta"]["new_h"]),
                "orig_w": float(targets[0]["resize_meta"]["orig_w"]),
                "orig_h": float(targets[0]["resize_meta"]["orig_h"]),
            }
            save_fused_image(outputs["fused"], resize_meta, save_path)


def main():
    args = parse_args()
    cfg = load_config(args.config)
    device = torch.device(args.device)

    model = build_model(cfg, device)
    load_checkpoint(model, args.checkpoint, device)

    if args.ir_path and args.vis_path:
        run_single_image(model, cfg, device, args)
    else:
        run_dataset_inference(model, cfg, device, args)


if __name__ == "__main__":
    main()
