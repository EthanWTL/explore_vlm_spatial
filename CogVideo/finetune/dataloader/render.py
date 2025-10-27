#!/usr/bin/env python3
# render.py
from __future__ import annotations

import math
from typing import Dict, List, Tuple, Optional

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch


# -----------------------------------------------------------------------------
# Geometry helpers (read from JSONL record)
# -----------------------------------------------------------------------------

def geom_from_rec(rec: Dict) -> Tuple[int, int, int, int, int, int, int, int]:
    """
    Extract exact geometry saved by maze_gen.py.

    Returns:
        rows, cols, H, W, cell_px, x0, y0, knob_px
    """
    rows    = int(rec["rows"])
    cols    = int(rec["cols"])
    W       = int(rec["canvas_w"])
    H       = int(rec["canvas_h"])
    cell_px = int(rec["cell_px"])
    x0      = int(rec["x0"])
    y0      = int(rec["y0"])
    knob_px = int(rec.get("knob_px", 14))  # default if missing
    return rows, cols, H, W, cell_px, x0, y0, knob_px


def cell_center_yx(r: float, c: float, cell_px: int, x0: int, y0: int) -> Tuple[float, float]:
    """Pixel center of cell (r,c) -> (y, x) in image coords."""
    x = x0 + (c + 0.5) * cell_px
    y = y0 + (r + 0.5) * cell_px
    return float(y), float(x)


# -----------------------------------------------------------------------------
# Basic I/O
# -----------------------------------------------------------------------------

def load_maze_png(image_dir: str, index: int) -> np.ndarray:
    """Load maze PNG (uint8 RGB) as (H,W,3). Filename: maze_{index:05d}.png"""
    from pathlib import Path
    path = Path(image_dir) / f"maze_{index:05d}.png"
    img = Image.open(path).convert("RGB")
    return np.array(img, dtype=np.uint8)


def img_u8_to_tensor_pm1(img_u8: np.ndarray) -> torch.FloatTensor:
    """uint8 (H,W,3) -> FloatTensor (3,H,W) in [-1,1]"""
    arr = img_u8.astype(np.float32) / 255.0
    arr = arr * 2.0 - 1.0
    arr = np.transpose(arr, (2, 0, 1))
    return torch.from_numpy(arr).float()


# -----------------------------------------------------------------------------
# Color utilities
# -----------------------------------------------------------------------------

def hsv_to_rgb_np(h: np.ndarray, s: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Vectorized HSV->RGB with H,S,V in [0,1]. Returns float RGB in [0,1], shape (...,3)."""
    i = np.floor(h * 6.0).astype(np.int32)
    f = h * 6.0 - i
    p = v * (1.0 - s)
    q = v * (1.0 - f * s)
    t = v * (1.0 - (1.0 - f) * s)
    i_mod = np.mod(i, 6)
    r = np.choose(i_mod, [v, q, p, p, t, v])
    g = np.choose(i_mod, [t, v, v, q, p, p])
    b = np.choose(i_mod, [p, p, t, v, v, q])
    return np.stack([r, g, b], axis=-1)


# -----------------------------------------------------------------------------
# Strict 49-frame sampler (arc length; exact endpoints)
# -----------------------------------------------------------------------------

def sample_positions_strict49(
    true_path: List[Tuple[int, int]],
    *,
    rows: int, cols: int,
    H: int, W: int,
    cell_px: int, x0: int, y0: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Evenly sample EXACTLY 49 positions along the polyline of cell centers.
    Returns (ys, xs) each of shape (49,). Endpoints are exact (up to rounding).
    """
    NUM = 49
    #assert len(true_path) >= 1

    def center(rc):
        r, c = rc
        return cell_center_yx(r, c, cell_px, x0, y0)

    if len(true_path) == 1:
        y, x = center(true_path[0])
        ys = np.full((NUM,), y, dtype=np.float32)
        xs = np.full((NUM,), x, dtype=np.float32)
        return ys, xs

    pts = np.asarray([center(rc) for rc in true_path], dtype=np.float32)  # (K,2) (y,x)
    seg_vecs = pts[1:] - pts[:-1]
    seg_len = np.linalg.norm(seg_vecs, axis=1)
    total = float(seg_len.sum())

    if total <= 1e-9:
        ys = np.full((NUM,), pts[0, 0], dtype=np.float32)
        xs = np.full((NUM,), pts[0, 1], dtype=np.float32)
        return ys, xs

    s_targets = np.linspace(0.0, total, NUM, dtype=np.float32)
    cum = np.concatenate([[0.0], np.cumsum(seg_len)])

    ys = np.empty((NUM,), dtype=np.float32)
    xs = np.empty((NUM,), dtype=np.float32)
    j = 0
    for k, s in enumerate(s_targets):
        while j + 1 < len(cum) and s > cum[j + 1]:
            j += 1
        if j >= len(seg_len):
            ys[k], xs[k] = pts[-1]
            continue
        seg_s = s - cum[j]
        alpha = 0.0 if seg_len[j] == 0 else (seg_s / seg_len[j])
        base = pts[j]
        ys[k] = base[0] + alpha * seg_vecs[j, 0]
        xs[k] = base[1] + alpha * seg_vecs[j, 1]

    ys = np.clip(ys, 0.0, H - 1.0)
    xs = np.clip(xs, 0.0, W - 1.0)
    ys[0], xs[0] = pts[0]
    ys[-1], xs[-1] = pts[-1]
    return ys, xs


# -----------------------------------------------------------------------------
# Flow video synthesis (strict 49 frames, HSV encoding)
# -----------------------------------------------------------------------------

def render_flow_hsv_video_strict49(
    true_path: List[List[int]] | List[Tuple[int, int]],
    rec_or_geom: Dict | Tuple[int, int, int, int, int, int, int, int],
    *,
    spot_radius_px: Optional[int] = None,      # if None, uses knob_px//2 from rec
    magnitude_norm: Optional[float] = None,    # if None, uses max |v|
    last_frame_zero_vel: bool = False,
    return_debug: bool = False,
) -> torch.FloatTensor | Tuple[torch.FloatTensor, Dict[str, np.ndarray]]:
    """
    Returns:
        video: (49,3,H,W) FloatTensor in [-1,1]
        (optionally, a debug dict with positions/vels)
    """
    # Unpack geometry
    if isinstance(rec_or_geom, dict):
        rows, cols, H, W, cell_px, x0, y0, knob_px = geom_from_rec(rec_or_geom)
        if spot_radius_px is None:
            spot_radius_px = max(1, knob_px // 2)
    else:
        rows, cols, H, W, cell_px, x0, y0, knob_px = rec_or_geom
        if spot_radius_px is None:
            spot_radius_px = max(1, knob_px // 2)

    path = [(int(r), int(c)) for (r, c) in true_path]

    ys, xs = sample_positions_strict49(
        path, rows=rows, cols=cols, H=H, W=W, cell_px=cell_px, x0=x0, y0=y0
    )

    vxs = np.zeros_like(xs, dtype=np.float32)
    vys = np.zeros_like(ys, dtype=np.float32)
    vxs[:-1] = xs[1:] - xs[:-1]
    vys[:-1] = ys[1:] - ys[:-1]
    if last_frame_zero_vel:
        vxs[-1] = 0.0
        vys[-1] = 0.0
    else:
        if len(xs) > 1:
            vxs[-1] = vxs[-2]
            vys[-1] = vys[-2]

    mags = np.sqrt(vxs * vxs + vys * vys)
    if magnitude_norm is None:
        denom = float(np.max(mags))
        magnitude_norm = denom if denom > 1e-6 else 1.0

    T = 49
    video_rgb = np.zeros((T, H, W, 3), dtype=np.float32)
    r2 = int(spot_radius_px) * int(spot_radius_px)

    for t in range(T):
        y0c = int(round(ys[t]))
        x0c = int(round(xs[t]))
        vx = float(vxs[t])
        vy = float(vys[t])

        hue = (math.atan2(vy, vx) + math.pi) / (2.0 * math.pi)  # [0,1]
        sat = float(np.clip(mags[t] / (magnitude_norm + 1e-6), 0.0, 1.0))
        val = 1.0

        y_min = max(0, y0c - spot_radius_px)
        y_max = min(H - 1, y0c + spot_radius_px)
        x_min = max(0, x0c - spot_radius_px)
        x_max = min(W - 1, x0c + spot_radius_px)

        ys_local = np.arange(y_min, y_max + 1, dtype=np.int32)[:, None]
        xs_local = np.arange(x_min, x_max + 1, dtype=np.int32)[None, :]
        dy = ys_local - y0c
        dx = xs_local - x0c
        mask = (dy * dy + dx * dx) <= r2
        if not np.any(mask):
            continue

        h = np.full(mask.shape, hue, dtype=np.float32)
        s = np.full(mask.shape, sat, dtype=np.float32)
        v = np.full(mask.shape, val, dtype=np.float32)
        rgb = hsv_to_rgb_np(h, s, v)  # (h,w,3) in [0,1]

        region = video_rgb[t, y_min:y_max + 1, x_min:x_max + 1, :]
        region[mask] = rgb[mask]
        video_rgb[t, y_min:y_max + 1, x_min:x_max + 1, :] = region

    video_rgb = video_rgb * 2.0 - 1.0
    video_rgb = np.transpose(video_rgb, (0, 3, 1, 2))
    video = torch.from_numpy(video_rgb).float()

    if return_debug:
        debug = dict(
            ys=ys, xs=xs, vxs=vxs, vys=vys, mags=mags,
            rows=rows, cols=cols, H=H, W=W, cell_px=cell_px, x0=x0, y0=y0,
            spot_radius_px=spot_radius_px, magnitude_norm=magnitude_norm,
        )
        return video, debug
    return video


def render_flow_from_record(
    rec: Dict,
    *,
    last_frame_zero_vel: bool = False,
    return_debug: bool = False,
) -> torch.FloatTensor | Tuple[torch.FloatTensor, Dict[str, np.ndarray]]:
    """Use rec['true_path'] + geometry to produce (49,3,H,W) in [-1,1]."""
    return render_flow_hsv_video_strict49(
        true_path=[tuple(rc) for rc in rec["true_path"]],
        rec_or_geom=rec,
        spot_radius_px=None,                 # use rec['knob_px']//2
        magnitude_norm=None,                 # normalize by max |v|
        last_frame_zero_vel=last_frame_zero_vel,
        return_debug=return_debug,
    )


# -----------------------------------------------------------------------------
# MP4 helpers (this includes save_flow_hsv_mp4 you’re importing)
# -----------------------------------------------------------------------------

def _video_tensor_to_uint8_frames(video: torch.Tensor) -> list[np.ndarray]:
    """(T,3,H,W) in [-1,1] -> list of (H,W,3) uint8"""
    assert video.ndim == 4 and video.shape[1] == 3
    v = video.detach().cpu().clamp(-1, 1)
    v = (v + 1.0) * 0.5
    v = (v * 255.0).round().byte()
    v = v.permute(0, 2, 3, 1).contiguous()
    return [frame.numpy() for frame in v]


def write_mp4(frames: list[np.ndarray], out_path: str, fps: int = 12) -> None:
    """Write a list of (H,W,3) uint8 frames to MP4 (cv2 -> imageio fallback)."""
    try:
        import cv2  # type: ignore
        h, w, _ = frames[0].shape
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
        if not writer.isOpened():
            raise RuntimeError("cv2.VideoWriter failed to open output")
        for f in frames:
            writer.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
        writer.release()
        return
    except Exception:
        pass
    try:
        import imageio.v3 as iio  # type: ignore
        iio.imwrite(out_path, frames, fps=fps, codec="libx264", quality=8)
        return
    except Exception as e:
        raise RuntimeError(
            "Could not save MP4 using cv2 or imageio. "
            "Install 'opencv-python' or 'imageio[pyav]'/'imageio-ffmpeg'. "
            f"Error: {e}"
        )


def save_flow_hsv_mp4(video: torch.Tensor, out_path: str, fps: int = 12) -> None:
    """Save flow video tensor (T,3,H,W) in [-1,1] to MP4."""
    frames = _video_tensor_to_uint8_frames(video)
    write_mp4(frames, out_path, fps=fps)


# -----------------------------------------------------------------------------
# Overlay helpers (handy for debugging alignment)
# -----------------------------------------------------------------------------

def draw_axes_background(
    rows: int, cols: int, H: int, W: int, cell_px: int, x0: int, y0: int,
    *, wall_px: int = 4, grid_gray: int = 220, grid_px: int = 1,
) -> np.ndarray:
    """Return uint8 RGB image with grid + outer border + row/col ticks."""
    im = Image.new("RGB", (W, H), (255, 255, 255))
    dr = ImageDraw.Draw(im)
    maze_w = cols * cell_px
    maze_h = rows * cell_px

    if grid_gray is not None:
        for cc in range(cols + 1):
            x = x0 + cc * cell_px
            dr.line([(x, y0), (x, y0 + maze_h)], fill=(grid_gray,)*3, width=grid_px)
        for rr in range(rows + 1):
            y = y0 + rr * cell_px
            dr.line([(x0, y), (x0 + maze_w, y)], fill=(grid_gray,)*3, width=grid_px)

    # Border
    dr.line([(x0, y0), (x0 + maze_w, y0)], fill=(0, 0, 0), width=wall_px)
    dr.line([(x0, y0 + maze_h), (x0 + maze_w, y0 + maze_h)], fill=(0, 0, 0), width=wall_px)
    dr.line([(x0, y0), (x0, y0 + maze_h)], fill=(0, 0, 0), width=wall_px)
    dr.line([(x0 + maze_w, y0), (x0 + maze_w, y0 + maze_h)], fill=(0, 0, 0), width=wall_px)

    # Ticks
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None
    for c in range(cols):
        cx = x0 + c * cell_px + cell_px // 2
        dr.text((cx - 3, max(0, y0 - 14)), str(c), fill=(0, 0, 0), font=font)
    dr.text((x0 + maze_w // 2 - 10, max(0, y0 - 28)), "col", fill=(0, 0, 0), font=font)

    for r in range(rows):
        cy = y0 + r * cell_px + cell_px // 2
        dr.text((max(0, x0 - 18), cy - 6), str(r), fill=(0, 0, 0), font=font)
    dr.text((max(0, x0 - 35), y0 + maze_h // 2 - 6), "row", fill=(0, 0, 0), font=font)

    return np.array(im, dtype=np.uint8)


def save_overlay_on_axes_mp4(
    video: torch.Tensor,
    rows: int, cols: int, H: int, W: int, cell_px: int, x0: int, y0: int,
    *, wall_px: int = 4, out_path: str = "overlay_axes.mp4", fps: int = 12,
) -> None:
    """Overlay flow frames on synthetic axes/grid background."""
    frames = _video_tensor_to_uint8_frames(video)
    bg = draw_axes_background(rows, cols, H, W, cell_px, x0, y0, wall_px=wall_px)

    comp_frames: list[np.ndarray] = []
    for f in frames:
        mask = (f.sum(axis=2) > 0)  # flow disc region
        comp = bg.copy()
        comp[mask] = f[mask]
        comp_frames.append(comp)

    write_mp4(comp_frames, out_path, fps=fps)


def save_overlay_on_png_mp4(video: torch.Tensor, png_u8: np.ndarray, out_path: str, fps: int = 12) -> None:
    """Overlay flow frames on top of a provided background PNG (uint8 HxWx3)."""
    frames = _video_tensor_to_uint8_frames(video)
    H, W, _ = frames[0].shape
    bg = Image.fromarray(png_u8).resize((W, H), Image.NEAREST)
    bg_u8 = np.array(bg, dtype=np.uint8)

    comp_frames: list[np.ndarray] = []
    for f in frames:
        mask = (f.sum(axis=2) > 0)
        comp = bg_u8.copy()
        comp[mask] = f[mask]
        comp_frames.append(comp)

    write_mp4(comp_frames, out_path, fps=fps)
