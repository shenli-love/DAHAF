import glob
import os
import xml.etree.ElementTree as ET

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


DEFAULT_CLASSES = {
    "People": 0,
    "Person": 0,
    "Car": 1,
    "Bus": 2,
    "Truck": 3,
    "Lamp": 4,
    "Motorcycle": 5,
}


def letterbox_resize(image, target_size, interpolation=cv2.INTER_AREA, pad_value=0):
    target_h, target_w = target_size
    if image.ndim == 3 and image.shape[2] == 1:
        image = image.squeeze(2)
    orig_h, orig_w = image.shape[:2]
    scale = min(target_w / float(orig_w), target_h / float(orig_h))
    new_w = max(1, int(round(orig_w * scale)))
    new_h = max(1, int(round(orig_h * scale)))
    resized = cv2.resize(image, (new_w, new_h), interpolation=interpolation)

    if image.ndim == 2:
        canvas = np.full((target_h, target_w), pad_value, dtype=resized.dtype)
    else:
        if resized.ndim == 2:
            resized = resized[..., None]
        canvas = np.full((target_h, target_w, image.shape[2]), pad_value, dtype=resized.dtype)

    pad_w = target_w - new_w
    pad_h = target_h - new_h
    pad_left = pad_w // 2
    pad_top = pad_h // 2
    canvas[pad_top:pad_top + new_h, pad_left:pad_left + new_w] = resized

    meta = {
        "scale": scale,
        "pad_left": pad_left,
        "pad_top": pad_top,
        "new_w": new_w,
        "new_h": new_h,
        "target_w": target_w,
        "target_h": target_h,
        "orig_w": orig_w,
        "orig_h": orig_h,
    }
    return canvas, meta


def remap_boxes_to_letterbox(boxes, meta):
    remapped = []
    scale = meta["scale"]
    pad_left = meta["pad_left"]
    pad_top = meta["pad_top"]
    for xmin, ymin, xmax, ymax in boxes:
        remapped.append([
            xmin * scale + pad_left,
            ymin * scale + pad_top,
            xmax * scale + pad_left,
            ymax * scale + pad_top,
        ])
    return remapped


def undo_letterbox_image(image, resize_meta, interpolation=cv2.INTER_LANCZOS4):
    pad_top = int(resize_meta["pad_top"])
    pad_left = int(resize_meta["pad_left"])
    new_h = int(resize_meta["new_h"])
    new_w = int(resize_meta["new_w"])
    orig_h = int(resize_meta["orig_h"])
    orig_w = int(resize_meta["orig_w"])
    cropped = image[pad_top:pad_top + new_h, pad_left:pad_left + new_w]
    return cv2.resize(cropped, (orig_w, orig_h), interpolation=interpolation)


class M3FDDataset(Dataset):
    def __init__(self, root, split="train", image_size=(256, 256), classes=None):
        self.root = root
        self.image_size = tuple(image_size)
        self.classes = classes or DEFAULT_CLASSES
        self.ir_dir = os.path.join(root, "ir")
        self.vis_dir = os.path.join(root, "vi")
        self.xml_dir = os.path.join(root, "Annotation")
        self.label_dir = os.path.join(root, "labels")
        self.meta_dir = os.path.join(root, "meta")
        self.samples = self._build_index(split)

        if not self.samples:
            raise RuntimeError(f"No samples found under {root}")

    def _build_index(self, split):
        meta_path = os.path.join(self.meta_dir, f"{split}.txt")
        if os.path.isfile(meta_path):
            with open(meta_path, "r", encoding="utf-8") as handle:
                names = [line.strip() for line in handle if line.strip()]
            return names

        ir_paths = glob.glob(os.path.join(self.ir_dir, "*.png")) + \
                   glob.glob(os.path.join(self.ir_dir, "*.jpg")) + \
                   glob.glob(os.path.join(self.ir_dir, "*.bmp"))

        names = sorted(os.path.splitext(os.path.basename(path))[0] for path in ir_paths)
        if not names:
            print(f"Warning: No images found in {self.ir_dir}")
            return []

        pivot = int(0.9 * len(names))
        if split == "train":
            return names[:pivot]
        if split == "val":
            return names[pivot:]
        return names

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        name = self.samples[idx]
        ir = cv2.imread(os.path.join(self.ir_dir, f"{name}.png"), cv2.IMREAD_GRAYSCALE)
        vis = cv2.imread(os.path.join(self.vis_dir, f"{name}.png"), cv2.IMREAD_COLOR)
        if ir is None or vis is None:
            raise RuntimeError(f"Failed to load pair {name}")

        vis_orig_ycrcb = cv2.cvtColor(vis, cv2.COLOR_BGR2YCrCb)
        vis_orig_y = vis_orig_ycrcb[:, :, 0].astype(np.float32) / 255.0
        vis_blur = cv2.GaussianBlur(vis_orig_y, (5, 5), sigmaX=1.0, sigmaY=1.0)
        vis_detail = ((vis_orig_y - vis_blur) + 1.0) * 0.5
        vis_detail = np.clip(vis_detail, 0.0, 1.0).astype(np.float32)

        ir, resize_meta = letterbox_resize(ir, self.image_size)
        vis, _ = letterbox_resize(vis, self.image_size)
        vis_detail, _ = letterbox_resize(
            vis_detail,
            self.image_size,
            interpolation=cv2.INTER_AREA,
            pad_value=0.5,
        )
        vis_ycrcb = cv2.cvtColor(vis, cv2.COLOR_BGR2YCrCb)
        vis_y, vis_cr, vis_cb = cv2.split(vis_ycrcb)

        target_h, target_w = vis_y.shape[:2]
        vis_cb = cv2.resize(vis_cb, (target_w, target_h), interpolation=cv2.INTER_AREA)
        vis_cr = cv2.resize(vis_cr, (target_w, target_h), interpolation=cv2.INTER_AREA)

        img_ir = torch.from_numpy(ir.astype(np.float32) / 255.0).unsqueeze(0)
        img_vis = torch.from_numpy(vis_y.astype(np.float32) / 255.0).unsqueeze(0)
        img_vis_chroma = torch.from_numpy(
            np.stack([vis_cb, vis_cr], axis=0).astype(np.float32) / 255.0
        )
        img_vis_detail = torch.from_numpy(vis_detail.astype(np.float32)).unsqueeze(0)

        boxes, labels = self._load_targets(name, resize_meta)
        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 4), dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.long) if labels else torch.zeros((0,), dtype=torch.long),
            "image_id": name,
            "vis_chroma": img_vis_chroma,
            "vis_detail": img_vis_detail,
            "orig_size": torch.tensor([resize_meta["orig_h"], resize_meta["orig_w"]], dtype=torch.float32),
            "resize_meta": {
                "scale": torch.tensor(resize_meta["scale"], dtype=torch.float32),
                "pad_left": torch.tensor(resize_meta["pad_left"], dtype=torch.float32),
                "pad_top": torch.tensor(resize_meta["pad_top"], dtype=torch.float32),
                "new_w": torch.tensor(resize_meta["new_w"], dtype=torch.float32),
                "new_h": torch.tensor(resize_meta["new_h"], dtype=torch.float32),
                "orig_w": torch.tensor(resize_meta["orig_w"], dtype=torch.float32),
                "orig_h": torch.tensor(resize_meta["orig_h"], dtype=torch.float32),
            },
        }
        return img_ir, img_vis, target

    def _load_targets(self, name, resize_meta):
        txt_path = os.path.join(self.label_dir, f"{name}.txt")
        xml_path = os.path.join(self.xml_dir, f"{name}.xml")
        if os.path.isfile(txt_path):
            return self._parse_yolo(txt_path, resize_meta)
        if os.path.isfile(xml_path):
            return self._parse_xml(xml_path, resize_meta)
        return [], []

    def _parse_yolo(self, path, resize_meta):
        boxes = []
        labels = []
        width = float(resize_meta["orig_w"])
        height = float(resize_meta["orig_h"])
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                cls_id, cx, cy, bw, bh = map(float, parts)
                x1 = (cx - bw / 2.0) * width
                y1 = (cy - bh / 2.0) * height
                x2 = (cx + bw / 2.0) * width
                y2 = (cy + bh / 2.0) * height
                boxes.append([x1, y1, x2, y2])
                labels.append(int(cls_id))
        return remap_boxes_to_letterbox(boxes, resize_meta), labels

    def _parse_xml(self, path, resize_meta):
        boxes = []
        labels = []
        root = ET.parse(path).getroot()
        for obj in root.findall("object"):
            cls_name = obj.findtext("name", default="")
            if cls_name not in self.classes:
                continue
            box = obj.find("bndbox")
            xmin = float(box.findtext("xmin", default="0"))
            ymin = float(box.findtext("ymin", default="0"))
            xmax = float(box.findtext("xmax", default="0"))
            ymax = float(box.findtext("ymax", default="0"))
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(self.classes[cls_name])
        return remap_boxes_to_letterbox(boxes, resize_meta), labels


def collate_fn(batch):
    img_ir = torch.stack([sample[0] for sample in batch], dim=0)
    img_vis = torch.stack([sample[1] for sample in batch], dim=0)
    targets = [sample[2] for sample in batch]
    return img_ir, img_vis, targets
