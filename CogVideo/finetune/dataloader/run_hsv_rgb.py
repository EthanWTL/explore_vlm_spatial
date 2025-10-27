#!/usr/bin/env python3
# run_hsv_rgb.py
from __future__ import annotations

import os
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from PIL import Image

# Our dataloader + renderer
from CogVideo.finetune.dataloader.maze_dataset import build_all_loaders
from CogVideo.finetune.dataloader.render import (
    load_maze_png,
    save_flow_hsv_mp4,
    save_overlay_on_png_mp4,
    save_overlay_on_axes_mp4,
    geom_from_rec,
    img_u8_to_tensor_pm1,
)

# -------------------------
# Hardcoded dataset paths
# -------------------------
TRAIN_JSONL = "/project/sds-rise/ethan/explore_diffusion_spatial/datasets/5by5/basic/train/info_labels.jsonl"
TRAIN_IMG   = "/project/sds-rise/ethan/explore_diffusion_spatial/datasets/5by5/basic/train/images"
TRAIN_PRM   = "/project/sds-rise/ethan/explore_diffusion_spatial/datasets/5by5/basic/train/prompts.txt"

VAL_JSONL   = "/project/sds-rise/ethan/explore_diffusion_spatial/datasets/5by5/basic/val/info_labels.jsonl"
VAL_IMG     = "/project/sds-rise/ethan/explore_diffusion_spatial/datasets/5by5/basic/val/images"
VAL_PRM     = "/project/sds-rise/ethan/explore_diffusion_spatial/datasets/5by5/basic/val/prompts.txt"

TEST_JSONL  = "/project/sds-rise/ethan/explore_diffusion_spatial/datasets/5by5/basic/test/info_labels.jsonl"
TEST_IMG    = "/project/sds-rise/ethan/explore_diffusion_spatial/datasets/5by5/basic/test/images"
TEST_PRM    = "/project/sds-rise/ethan/explore_diffusion_spatial/datasets/5by5/basic/test/prompts.txt"

# -------------------------
# Output and runtime opts
# -------------------------
OUT_DIR      = Path("debug_out")
BATCH_SIZE   = 1
NUM_WORKERS  = 0
NUM_FRAMES   = 49     # must be 49 for strict renderer
SAVE_FIRST_K = 3
FPS          = 12

SPLIT = "train"       # choose: "train" | "val" | "test"


# -------------------------
# Helpers
# -------------------------

def _tensor_img_pm1_to_u8(img: torch.Tensor) -> np.ndarray:
    """(3,H,W) in [-1,1] -> (H,W,3) uint8 RGB"""
    assert img.ndim == 3 and img.shape[0] == 3, f"expected (3,H,W), got {tuple(img.shape)}"
    x = img.detach().cpu().clamp(-1, 1)
    x = (x + 1.0) * 0.5
    x = (x * 255.0).round().byte()
    x = x.permute(1, 2, 0).contiguous().numpy()
    return x

def save_png_from_img_tensor(img: torch.Tensor, out_path: str) -> None:
    """Save (3,H,W) in [-1,1] to PNG."""
    u8 = _tensor_img_pm1_to_u8(img)
    Image.fromarray(u8).save(out_path)


# -------------------------
# Main
# -------------------------

def main() -> None:
    torch.set_grad_enabled(False)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    (train_ds, train_dl), (val_ds, val_dl), (test_ds, test_dl) = build_all_loaders(
        train_jsonl=TRAIN_JSONL, train_images=TRAIN_IMG, train_prompts=TRAIN_PRM,
        val_jsonl=VAL_JSONL,     val_images=VAL_IMG,     val_prompts=VAL_PRM,
        test_jsonl=TEST_JSONL,   test_images=TEST_IMG,   test_prompts=TEST_PRM,
        batch_size=BATCH_SIZE, num_workers=NUM_WORKERS,
        num_frames=NUM_FRAMES, assert_geometry_match=True, shuffle_train=True
    )

    if SPLIT == "train":
        ds, dl = train_ds, train_dl
    elif SPLIT == "val":
        ds, dl = val_ds, val_dl
    else:
        ds, dl = test_ds, test_dl

    if ds is None or dl is None:
        raise RuntimeError(f"{SPLIT} split not available")

    batch = next(iter(dl))

    images  = batch["image"]      # (B,3,H,W) in [-1,1]
    flows   = batch["flow_hsv"]   # (B,49,3,H,W) in [-1,1]
    prompts = batch["prompt"]     # list[str]
    indices = batch["index"]      # (B,)
    recs    = batch["rec"]        # list[dict]

    print(f"image.shape: {tuple(images.shape)}")
    print(f"flow_hsv.shape: {tuple(flows.shape)}")
    print(f"#prompts: {len(prompts)}  indices: {indices.tolist()}")

    # Quick sanity on ranges
    img_min, img_max = float(images.min().item()), float(images.max().item())
    flow_min, flow_max = float(flows.min().item()), float(flows.max().item())
    print(f"Image range: [{img_min:.3f}, {img_max:.3f}] (expect ~[-1,1])")
    print(f"Flow  range: [{flow_min:.3f}, {flow_max:.3f}] (expect ~[-1,1])")

    B = images.shape[0]
    K = min(SAVE_FIRST_K, B)

    for b in range(K):
        idx = int(indices[b])
        rec = recs[b]
        rows, cols, H, W, cell_px, x0, y0, knob_px = (
            int(rec["rows"]), int(rec["cols"]), int(rec["canvas_h"]), int(rec["canvas_w"]),
            int(rec["cell_px"]), int(rec["x0"]), int(rec["y0"]), int(rec.get("knob_px", 14))
        )

        # File paths
        base = f"{SPLIT}_{idx:05d}"
        png_out   = OUT_DIR / f"{base}_img.png"
        flow_out  = OUT_DIR / f"{base}_flow.mp4"
        over_png  = OUT_DIR / f"{base}_overlay_on_png.mp4"
        over_axes = OUT_DIR / f"{base}_overlay_on_axes.mp4"

        # Save the (reloaded) image PNG for reference
        save_png_from_img_tensor(images[b], str(png_out))

        # Save the raw flow HSV mp4
        save_flow_hsv_mp4(flows[b], str(flow_out), fps=FPS)

        # Overlays for debugging alignment
        # 1) Overlay on the original PNG
        bg_png_u8 = load_maze_png(TRAIN_IMG if SPLIT=="train" else VAL_IMG if SPLIT=="val" else TEST_IMG, idx)
        save_overlay_on_png_mp4(flows[b], bg_png_u8, str(over_png), fps=FPS)

        # 2) Overlay on synthetic axes background (uses exact geometry)
        save_overlay_on_axes_mp4(
            flows[b],
            rows, cols, H, W, cell_px, x0, y0,
            wall_px=int(rec.get("wall_px", 4)),
            out_path=str(over_axes),
            fps=FPS,
        )

        print(f"Saved:\n  {png_out}\n  {flow_out}\n  {over_png}\n  {over_axes}")

    print("Done.")

if __name__ == "__main__":
    main()
