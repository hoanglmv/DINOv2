"""Kiểm tra nhanh các hàm tô màu / overlay.

Chạy:
    cd src
    python -m zero_shot_seg.test.test_visualize
"""

import numpy as np
import torch

from zero_shot_seg.visualize import (
    DEFAULT_PALETTE,
    labels_to_color,
    legend_entries,
    overlay_segmentation,
)


H, W = 32, 48


def test_labels_to_color_numpy():
    labels = np.zeros((H, W), dtype=np.int64)
    labels[:, W // 2 :] = 1
    color = labels_to_color(labels)
    assert color.shape == (H, W, 3) and color.dtype == np.uint8
    assert tuple(color[0, 0]) == tuple(DEFAULT_PALETTE[0])
    assert tuple(color[0, W // 2]) == tuple(DEFAULT_PALETTE[1])
    print("ok: labels_to_color (numpy input).")


def test_labels_to_color_tensor():
    labels = torch.zeros(H, W, dtype=torch.long)
    labels[H // 2 :, :] = 2
    color = labels_to_color(labels)
    assert color.shape == (H, W, 3)
    assert tuple(color[-1, -1]) == tuple(DEFAULT_PALETTE[2])
    print("ok: labels_to_color (tensor input).")


def test_overlay_segmentation():
    img = np.full((H, W, 3), 128, dtype=np.uint8)
    labels = np.zeros((H, W), dtype=np.int64)
    out = overlay_segmentation(img, labels, alpha=0.5)
    assert out.shape == img.shape and out.dtype == np.uint8
    # 0.5 * 128 + 0.5 * palette[0] — gần với trung bình.
    expected = (0.5 * 128 + 0.5 * DEFAULT_PALETTE[0]).astype(np.uint8)
    assert np.all(np.abs(out[0, 0].astype(int) - expected.astype(int)) <= 1)
    print("ok: overlay_segmentation tính đúng giá trị blend.")


def test_overlay_size_mismatch_raises():
    img = np.zeros((H, W, 3), dtype=np.uint8)
    labels = np.zeros((H + 1, W), dtype=np.int64)
    try:
        overlay_segmentation(img, labels)
    except ValueError:
        print("ok: từ chối kích thước không khớp.")
        return
    raise AssertionError("Phải raise ValueError.")


def test_legend_entries():
    names = ["a", "b", "c"]
    entries = legend_entries(names)
    assert len(entries) == 3
    assert entries[0] == ("a", tuple(int(x) for x in DEFAULT_PALETTE[0]))
    print("ok: legend_entries.")


if __name__ == "__main__":
    test_labels_to_color_numpy()
    test_labels_to_color_tensor()
    test_overlay_segmentation()
    test_overlay_size_mismatch_raises()
    test_legend_entries()
    print("\nAll visualize tests passed.")
