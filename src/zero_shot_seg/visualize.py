"""Tô màu segmentation map và overlay lên ảnh gốc."""

from typing import Optional, Sequence

import numpy as np
import torch

# Bảng màu cho tối đa 10 class — đủ dùng cho demo, có thể mở rộng nếu cần.
DEFAULT_PALETTE = np.array([
    [220,  20,  60],   # 0 — crimson
    [ 70, 130, 180],   # 1 — steelblue
    [  0, 200,   0],   # 2 — green
    [255, 165,   0],   # 3 — orange
    [128,   0, 128],   # 4 — purple
    [255, 215,   0],   # 5 — gold
    [  0, 200, 200],   # 6 — cyan
    [255, 105, 180],   # 7 — pink
    [128, 128, 128],   # 8 — gray
    [  0,   0,   0],   # 9 — black
], dtype=np.uint8)


def labels_to_color(labels, palette: np.ndarray = DEFAULT_PALETTE) -> np.ndarray:
    """Map label map (H, W) → ảnh RGB (H, W, 3) uint8."""
    if isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().numpy()
    labels = labels.astype(np.int64)
    return palette[labels % palette.shape[0]]


def overlay_segmentation(
    image_rgb_uint8: np.ndarray,
    labels,
    alpha: float = 0.5,
    palette: np.ndarray = DEFAULT_PALETTE,
) -> np.ndarray:
    """Trộn ảnh gốc (đã denormalize, uint8) với màu segmentation."""
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha phải trong [0, 1].")
    color = labels_to_color(labels, palette)
    if color.shape[:2] != image_rgb_uint8.shape[:2]:
        raise ValueError(
            f"Kích thước không khớp: image={image_rgb_uint8.shape[:2]} vs labels={color.shape[:2]}"
        )
    blended = image_rgb_uint8.astype(np.float32) * (1 - alpha) + color.astype(np.float32) * alpha
    return np.clip(blended, 0, 255).astype(np.uint8)


def legend_entries(class_names: Sequence[str], palette: np.ndarray = DEFAULT_PALETTE):
    """Sinh list (name, rgb_tuple) phục vụ matplotlib legend."""
    return [(name, tuple(int(c) for c in palette[i % palette.shape[0]]))
            for i, name in enumerate(class_names)]
