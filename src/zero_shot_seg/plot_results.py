"""Sinh figures cho báo cáo §2.2 từ kết quả batch eval.

Đọc:
    - `logs/zero_shot_seg_results.json`  (output của evaluate.py)
    - `logs/seg_samples/<name>/{original.png, gt.png, <backbone>_<method>.png}`

Sinh ra (mặc định vào `report/figures/`):
    - seg_miou_compare.png
    - seg_per_class_iou.png
    - seg_qualitative.png
    - seg_clustering_emergence.png
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from .classes import OXFORD_PET_TRIMAP_CLASSES
from .visualize import DEFAULT_PALETTE, overlay_segmentation


BACKBONE_DISPLAY = {
    "dinov2_vits14": "DINOv2-S",
    "dinov2_vitb14": "DINOv2-B",
    "vit_b_16_in1k": "ViT-B/16 (IN-1K)",
}
METHOD_DISPLAY = {
    "prototype": "Prototype",
    "cluster": "Clustering",
}


def _load_label_png(path: Path) -> np.ndarray:
    return np.array(Image.open(path))


def plot_miou_compare(results: dict, out_path: Path) -> None:
    backbones = results["config"]["backbones"]
    methods = results["config"]["methods"]
    data = results["results"]

    width = 0.35
    x = np.arange(len(backbones))
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for i, method in enumerate(methods):
        vals = [data[b][method]["mIoU"] * 100 for b in backbones]
        bars = ax.bar(x + (i - (len(methods) - 1) / 2) * width, vals, width,
                       label=METHOD_DISPLAY.get(method, method))
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                     f"{v:.1f}", ha="center", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels([BACKBONE_DISPLAY.get(b, b) for b in backbones])
    ax.set_ylabel("mIoU (%)")
    ax.set_title("Zero-shot Segmentation trên Oxford-IIIT Pet")
    ax.set_ylim(0, max(70, max(v + 5 for b in backbones for m in methods
                                  for v in [data[b][m]["mIoU"] * 100])))
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_per_class_iou(results: dict, out_path: Path, method: str = "prototype") -> None:
    backbones = results["config"]["backbones"]
    classes = results["config"]["class_names"]
    data = results["results"]

    width = 0.8 / len(backbones)
    x = np.arange(len(classes))
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for i, b in enumerate(backbones):
        per = data[b][method]["per_class_iou"]
        vals = [(per.get(c) or 0.0) * 100 for c in classes]
        bars = ax.bar(x + (i - (len(backbones) - 1) / 2) * width, vals, width,
                       label=BACKBONE_DISPLAY.get(b, b))
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                     f"{v:.1f}", ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(classes)
    ax.set_ylabel("IoU (%)")
    ax.set_title(f"IoU theo class — phương pháp {METHOD_DISPLAY.get(method, method)}")
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _overlay_from_label_file(rgb: np.ndarray, label_path: Path) -> np.ndarray:
    label = _load_label_png(label_path)
    return overlay_segmentation(rgb, label)


def _class_legend_handles(class_names=None):
    """Tạo handles legend màu → tên class theo OXFORD_PET_TRIMAP_CLASSES."""
    class_names = class_names or OXFORD_PET_TRIMAP_CLASSES
    handles = []
    for i, name in enumerate(class_names):
        rgb = DEFAULT_PALETTE[i % DEFAULT_PALETTE.shape[0]] / 255.0
        handles.append(mpatches.Patch(color=tuple(rgb), label=name))
    return handles


def plot_qualitative(samples_dir: Path, results: dict, out_path: Path,
                      max_rows: int = 4) -> None:
    """Grid (rows ảnh) x (cols = original | GT | DINOv2-B proto | DINOv2-B cluster | ViT-B/16 proto)."""
    order = json.loads((samples_dir / "sample_order.json").read_text())
    names = order["names"][:max_rows]

    columns = [
        ("Original", None),
        ("Ground truth", ("gt",)),
        ("DINOv2-B / Prototype", ("dinov2_vitb14", "prototype")),
        ("DINOv2-B / Clustering", ("dinov2_vitb14", "cluster")),
        ("ViT-B/16 / Prototype", ("vit_b_16_in1k", "prototype")),
    ]
    fig, axes = plt.subplots(len(names), len(columns),
                              figsize=(3 * len(columns), 3 * len(names)))
    if len(names) == 1:
        axes = np.array([axes])

    for r, name in enumerate(names):
        sd = samples_dir / name
        rgb = _load_label_png(sd / "original.png")
        # Đảm bảo rgb là (H, W, 3).
        if rgb.ndim == 2:
            rgb = np.stack([rgb] * 3, axis=-1)
        for c, (title, key) in enumerate(columns):
            ax = axes[r, c]
            if key is None:
                img = rgb
            elif key == ("gt",):
                img = overlay_segmentation(rgb, _load_label_png(sd / "gt.png"))
            else:
                backbone, method = key
                img = _overlay_from_label_file(rgb, sd / f"{backbone}_{method}.png")
            ax.imshow(img)
            ax.set_xticks([])
            ax.set_yticks([])
            if r == 0:
                ax.set_title(title, fontsize=10)
            if c == 0:
                miou = order["miou"].get(name, 0.0) * 100
                ax.set_ylabel(f"{name}\n(mIoU={miou:.1f}%)", fontsize=8)

    fig.suptitle("So sánh định tính — Zero-shot Segmentation", fontsize=12)
    fig.legend(
        handles=_class_legend_handles(),
        loc="lower center", ncol=len(OXFORD_PET_TRIMAP_CLASSES),
        bbox_to_anchor=(0.5, -0.01), frameon=False, fontsize=10,
    )
    fig.tight_layout(rect=(0.0, 0.03, 1.0, 0.97))
    fig.savefig(out_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def plot_clustering_emergence(samples_dir: Path, out_path: Path,
                               backbones=("dinov2_vitb14", "vit_b_16_in1k"),
                               max_rows: int = 3) -> None:
    """So sánh KMeans trên patch features của các backbone (SSL vs supervised)."""
    order = json.loads((samples_dir / "sample_order.json").read_text())
    names = order["names"][:max_rows]
    ncols = 1 + len(backbones)

    fig, axes = plt.subplots(len(names), ncols,
                              figsize=(3 * ncols, 3 * len(names)))
    if len(names) == 1:
        axes = np.array([axes])

    for r, name in enumerate(names):
        sd = samples_dir / name
        rgb = _load_label_png(sd / "original.png")
        if rgb.ndim == 2:
            rgb = np.stack([rgb] * 3, axis=-1)
        axes[r, 0].imshow(rgb)
        axes[r, 0].set_xticks([]); axes[r, 0].set_yticks([])
        if r == 0:
            axes[r, 0].set_title("Ảnh gốc", fontsize=10)

        for c, backbone in enumerate(backbones, start=1):
            cluster_label = _load_label_png(sd / f"{backbone}_cluster.png")
            axes[r, c].imshow(overlay_segmentation(rgb, cluster_label))
            axes[r, c].set_xticks([]); axes[r, c].set_yticks([])
            if r == 0:
                axes[r, c].set_title(
                    f"KMeans / {BACKBONE_DISPLAY.get(backbone, backbone)}", fontsize=10
                )

    fig.suptitle("Object-centric emergence từ KMeans — DINOv2 (SSL) vs ViT (IN-1K supervised)",
                  fontsize=11)
    fig.legend(
        handles=_class_legend_handles(),
        loc="lower center", ncol=len(OXFORD_PET_TRIMAP_CLASSES),
        bbox_to_anchor=(0.5, -0.01), frameon=False, fontsize=10,
    )
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(out_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Sinh figures báo cáo §2.2.")
    parser.add_argument("--results", type=str, default="../logs/zero_shot_seg_results.json")
    parser.add_argument("--samples", type=str, default="../logs/seg_samples")
    parser.add_argument("--out", type=str, default="../report/figures")
    args = parser.parse_args()

    results = json.loads(Path(args.results).read_text())
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    samples_dir = Path(args.samples)

    plot_miou_compare(results, out_dir / "seg_miou_compare.png")
    print(f"→ {out_dir / 'seg_miou_compare.png'}")

    plot_per_class_iou(results, out_dir / "seg_per_class_iou.png", method="prototype")
    print(f"→ {out_dir / 'seg_per_class_iou.png'}")

    plot_qualitative(samples_dir, results, out_dir / "seg_qualitative.png")
    print(f"→ {out_dir / 'seg_qualitative.png'}")

    plot_clustering_emergence(samples_dir, out_dir / "seg_clustering_emergence.png")
    print(f"→ {out_dir / 'seg_clustering_emergence.png'}")


if __name__ == "__main__":
    main()
