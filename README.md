# DAHAF-Net Lite

DAHAF-Net Lite is a lightweight infrared-visible image fusion project built around a single-path fusion pipeline.

## Active Pipeline

```text
IR / VIS
  -> Dual Encoder
  -> HCFB Lite Fusion Core
  -> Edge-Refined U-Decoder
  -> Fused Image
  -> Frozen Detector (single pass)
  -> Objectness-Only Guidance
  -> Final Fused Output
```

## Training

```bash
python scripts/train.py --config configs/default.yaml
```

## Inference

```bash
python infer.py --checkpoint checkpoints/best.pth
```

Single-image inference:

```bash
python infer.py --checkpoint checkpoints/best.pth --ir-path path/to/ir.png --vis-path path/to/vis.png
```

## Project Layout

```text
models/
  de_encoder.py
  hcfb.py
  decoder.py
  yolo11_bridge.py
  dahaf_net.py
utils/
  dataset.py
  loss.py
scripts/
  train.py
configs/
  default.yaml
infer.py
analyze_model.py
```

## Current Design Notes

- Encoder channels: `32 / 64 / 128`
- Decoder: edge-refined U-Net/FPN style
- Task guidance: objectness-only
- Detection branch: single frozen detector pass
- Fusion loss: intensity + gradient + edge-aware + SSIM + weak objectness auxiliary

