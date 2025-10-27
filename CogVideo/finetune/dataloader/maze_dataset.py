#!/usr/bin/env python3
# maze_dataset.py
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from PIL import Image

# Import rendering + image helpers from the new render.py
from CogVideo.finetune.dataloader.render import (
    load_maze_png,
    img_u8_to_tensor_pm1,             # uint8 (H,W,3) -> (3,H,W) in [-1,1]
    render_flow_from_record,          # builds (49,3,H,W) in [-1,1] using exact geometry
)


class MazeDataset(Dataset):
    """
    Each item returns:
      - image: FloatTensor (3,H,W) in [-1,1]   (the PNG drawn by maze_gen.py)
      - prompt: str
      - flow_hsv: FloatTensor (49,3,H,W) in [-1,1]  (strict 49 frames)
      - index: int  (rec["index"])
      - rec: dict   (full JSONL record, in case you need geometry downstream)
    """
    def __init__(
        self,
        jsonl_path: str,
        image_dir: str,
        prompts_path: str,
        *,
        num_frames: int = 49,                         # must be 49 for the strict renderer
        assert_geometry_match: bool = True,           # sanity-check PNG vs JSONL geometry
    ):
        super().__init__()
        self.jsonl_path = Path(jsonl_path)
        self.image_dir  = Path(image_dir)
        self.prompts_path = Path(prompts_path)
        self.num_frames = num_frames
        if self.num_frames != 49:
            raise ValueError("This dataset expects num_frames=49 to match the strict renderer.")

        # Load JSONL records
        self.records: List[Dict] = []
        with self.jsonl_path.open("r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                self.records.append(json.loads(line))

        # Load prompts (1:1 with records)
        with self.prompts_path.open("r") as f:
            self.prompts = [ln.rstrip("\n") for ln in f]

        if len(self.prompts) != len(self.records):
            raise ValueError(
                f"prompts count ({len(self.prompts)}) != records count ({len(self.records)}). "
                "They must align one-to-one (same order)."
            )

        self.assert_geometry_match = assert_geometry_match

    def __len__(self) -> int:
        return len(self.records)

    def _read_image_tensor(self, index: int, rec: Dict) -> torch.FloatTensor:
        """
        Load the PNG and convert to (3,H,W) FloatTensor in [-1,1].
        Optionally asserts the PNG size matches rec["canvas_h"/"canvas_w"].
        """
        img_u8 = load_maze_png(str(self.image_dir), index)  # (H,W,3) uint8
        H_png, W_png, _ = img_u8.shape

        if self.assert_geometry_match:
            H_rec = int(rec["canvas_h"])
            W_rec = int(rec["canvas_w"])
            if (H_png, W_png) != (H_rec, W_rec):
                raise AssertionError(
                    f"PNG size {(H_png, W_png)} != record canvas {(H_rec, W_rec)} for index {index}"
                )

        return img_u8_to_tensor_pm1(img_u8)

    def _render_flow_tensor(self, rec: Dict) -> torch.FloatTensor:
        """
        Use geometry + true_path from the record to create (49,3,H,W) in [-1,1].
        """
        out = render_flow_from_record(rec, last_frame_zero_vel=False, return_debug=False)
        # out is a FloatTensor (49,3,H,W)
        return out

    def __getitem__(self, i: int) -> Dict:
        rec = self.records[i]
        idx = int(rec["index"])
        prompt = self.prompts[i]

        # Image tensor (3,H,W)
        img = self._read_image_tensor(idx, rec)

        # Flow HSV video tensor (49,3,H,W)
        flow = self._render_flow_tensor(rec)

        return {
            "image": img,
            "prompt": prompt,
            "flow_hsv": flow,
            "index": idx,
            "rec": rec,
        }


# -------------------------
# Collate + loader builder
# -------------------------

def default_collate_fn(batch: List[Dict]) -> Dict:
    """
    Collate a list of dicts with keys:
      image: (3,H,W) FloatTensor
      flow_hsv: (49,3,H,W) FloatTensor (same H/W across batch)
      prompt: str
      index: int
      rec: dict
    """
    images = torch.stack([b["image"] for b in batch], dim=0)           # (B,3,H,W)
    flows  = torch.stack([b["flow_hsv"] for b in batch], dim=0)        # (B,49,3,H,W)
    prompts = [b["prompt"] for b in batch]
    indices = torch.tensor([b["index"] for b in batch], dtype=torch.long)
    recs    = [b["rec"] for b in batch]
    return {
        "image": images,
        "flow_hsv": flows,
        "prompt": prompts,
        "index": indices,
        "rec": recs,
    }


def build_all_loaders(
    *,
    train_jsonl: Optional[str] = None,
    train_images: Optional[str] = None,
    train_prompts: Optional[str] = None,
    val_jsonl: Optional[str] = None,
    val_images: Optional[str] = None,
    val_prompts: Optional[str] = None,
    test_jsonl: Optional[str] = None,
    test_images: Optional[str] = None,
    test_prompts: Optional[str] = None,
    batch_size: int = 4,
    num_workers: int = 4,
    num_frames: int = 49,
    assert_geometry_match: bool = True,
    shuffle_train: bool = True,
):
    """
    Returns tuples: (train_ds, train_dl), (val_ds, val_dl), (test_ds, test_dl)
    Any split with missing paths returns (None, None).
    """

    def make_ds(jsonl_path, image_dir, prompts_path, shuffle: bool):
        if not (jsonl_path and image_dir and prompts_path):
            return None, None
        ds = MazeDataset(
            jsonl_path=jsonl_path,
            image_dir=image_dir,
            prompts_path=prompts_path,
            num_frames=num_frames,
            assert_geometry_match=assert_geometry_match,
        )
        dl = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=default_collate_fn,
            drop_last=False,
        )
        return ds, dl

    train = make_ds(train_jsonl, train_images, train_prompts, shuffle_train)
    val   = make_ds(val_jsonl,   val_images,   val_prompts,   False)
    test  = make_ds(test_jsonl,  test_images,  test_prompts,  False)

    return train, val, test
