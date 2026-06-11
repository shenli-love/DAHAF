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
    image, resize_meta = letterbox_resize(image, image_size)
    tensor = torch.from_numpy(image.astype(np.float32) / 255.0).unsqueeze(0).unsqueeze(0).to(device)
    return tensor, resize_meta


def load_single_visible_ycbcr(path, image_size, device):
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Failed to load image: {path}")
    orig_ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    orig_y = orig_ycrcb[:, :, 0].astype(np.float32) / 255.0
    blur = cv2.GaussianBlur(orig_y, (5, 5), sigmaX=1.0, sigmaY=1.0)
    detail = np.abs(orig_y - blur)
    detail = np.clip(detail * 4.0, 0.0, 1.0).astype(np.float32)

    image, resize_meta = letterbox_resize(image, image_size)
    detail, _ = letterbox_resize(detail, image_size, interpolation=cv2.INTER_AREA, pad_value=0.5)
    ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycrcb)
    img_vis = torch.from_numpy(y.astype(np.float32) / 255.0).unsqueeze(0).unsqueeze(0).to(device)
    img_vis_chroma = torch.from_numpy(
        np.stack([cb, cr], axis=0).astype(np.float32) / 255.0
    ).unsqueeze(0).to(device)
    img_vis_detail = torch.from_numpy(detail.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)
    return img_vis, img_vis_chroma, img_vis_detail, resize_meta


def ycbcr_to_rgb(y, cb, cr):
    y = y.astype(np.float32)
    cb = cb.astype(np.float32) - 0.5
    cr = cr.astype(np.float32) - 0.5
    r = y + 1.402 * cr
    g = y - 0.344136 * cb - 0.714136 * cr
    b = y + 1.772 * cb
    return np.stack([r, g, b], axis=-1).clip(0.0, 1.0)


def _restore_chroma(chroma, chroma_meta, target_shape):
    cb = chroma[0]
    cr = chroma[1]
    cb = undo_letterbox_image(cb, chroma_meta, interpolation=cv2.INTER_CUBIC)
    cr = undo_letterbox_image(cr, chroma_meta, interpolation=cv2.INTER_CUBIC)
    target_h, target_w = target_shape
    if cb.shape[:2] != (target_h, target_w):
        cb = cv2.resize(cb, (target_w, target_h), interpolation=cv2.INTER_CUBIC)
        cr = cv2.resize(cr, (target_w, target_h), interpolation=cv2.INTER_CUBIC)
    return cb.clip(0.0, 1.0), cr.clip(0.0, 1.0)


def apply_clahe(image, clip_limit=1.2, tile_grid_size=(8, 8)):
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return clahe.apply(image)


def save_fused_image(fused, resize_meta, save_path, chroma=None, chroma_meta=None):
    fused_y = fused.squeeze().cpu().numpy()
    fused_y = undo_letterbox_image(fused_y, resize_meta, interpolation=cv2.INTER_LANCZOS4)
    fused_y = fused_y.clip(0.0, 1.0)
    image = (fused_y * 255.0).clip(0, 255).astype("uint8")
    cv2.imwrite(save_path, image)


def run_single_image(model, cfg, device, args):
    image_size = tuple(cfg["DATA"]["image_size"])
    img_ir, resize_meta = load_single_grayscale(args.ir_path, image_size, device)
    img_vis, img_vis_chroma, img_vis_detail, chroma_meta = load_single_visible_ycbcr(
        args.vis_path,
        image_size,
        device,
    )

    targets = [{
        "boxes": torch.zeros((0, 4), dtype=torch.float32),
        "labels": torch.zeros((0,), dtype=torch.long),
        "image_id": args.output_name,
        "vis_chroma": img_vis_chroma.squeeze(0).cpu(),
        "vis_detail": img_vis_detail.squeeze(0).cpu(),
        "orig_size": torch.tensor([resize_meta["orig_h"], resize_meta["orig_w"]], dtype=torch.float32),
        "resize_meta": {
            key: torch.tensor(value, dtype=torch.float32)
            for key, value in resize_meta.items()
            if key in {"scale", "pad_left", "pad_top", "new_w", "new_h", "orig_w", "orig_h"}
        },
    }]

    with torch.no_grad():
        outputs = model(
            img_ir,
            img_vis,
            img_vis_chroma=img_vis_chroma,
            detail_map=img_vis_detail,
            targets=targets,
            run_detection=args.with_det,
        )

    os.makedirs(args.output, exist_ok=True)
    save_path = os.path.join(args.output, f"{args.output_name}.png")
    save_fused_image(outputs["fused"], resize_meta, save_path, chroma=img_vis_chroma, chroma_meta=chroma_meta)
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
            img_vis_chroma = None
            if targets and "vis_chroma" in targets[0]:
                img_vis_chroma = torch.stack([target["vis_chroma"] for target in targets], dim=0).to(device)
            img_vis_detail = None
            if targets and "vis_detail" in targets[0]:
                img_vis_detail = torch.stack([target["vis_detail"] for target in targets], dim=0).to(device)
            outputs = model(
                img_ir,
                img_vis,
                img_vis_chroma=img_vis_chroma,
                detail_map=img_vis_detail,
                targets=targets,
                run_detection=args.with_det,
            )

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
            save_fused_image(outputs["fused"], resize_meta, save_path, chroma=img_vis_chroma, chroma_meta=resize_meta)


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
