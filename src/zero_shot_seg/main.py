"""CLI chạy zero-shot segmentation trên Oxford-IIIT Pet.

Cách chạy (từ thư mục src/):
    python -m zero_shot_seg.main --data_root ../data/oxford_pet \
        --method prototype --num_references 3 --num_samples 5 --device cpu

Trên CPU, hãy giữ image_size nhỏ (224) và dùng dinov2_vits14 để chạy được trong vài giây/ảnh.
"""

import argparse
from pathlib import Path

import numpy as np
import torch

from .classes import OXFORD_PET_TRIMAP_CLASSES
from .data import OxfordPetSegDataset, denormalize
from .feature_extractor import DINOv2DenseExtractor
from .metrics import segmentation_metrics
from .segmenter import ClusteringSegmenter, PrototypeZeroShotSegmenter
from .visualize import labels_to_color, overlay_segmentation


def mean_iou(pred: torch.Tensor, target: torch.Tensor, num_classes: int) -> float:
    """Wrapper giữ tương thích CLI cũ — gọi `metrics.segmentation_metrics`."""
    return segmentation_metrics(pred, target, num_classes)["mIoU"]


def run_prototype(args, extractor, dataset):
    classes = OXFORD_PET_TRIMAP_CLASSES
    references = {c: [] for c in classes}

    # Lấy num_references ảnh đầu tiên làm reference, dùng trimap mask để
    # chọn đúng patch của từng class.
    for i in range(args.num_references):
        img, mask, _ = dataset[i]
        for cls_idx, cls in enumerate(classes):
            cls_mask = mask == cls_idx
            if cls_mask.sum() > 0:
                references[cls].append((img, cls_mask))

    segmenter = PrototypeZeroShotSegmenter(extractor, classes)
    segmenter.build_prototypes(references)

    start = args.num_references
    stop = start + args.num_samples
    out_dir = Path(args.output_dir) if args.output_dir else None
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)

    mious = []
    for i in range(start, stop):
        img, mask, name = dataset[i]
        pred = segmenter.segment(img).cpu()
        miou = mean_iou(pred, mask, len(classes))
        mious.append(miou)
        print(f"[prototype] {name}: mIoU={miou:.3f}")

        if out_dir is not None:
            from PIL import Image
            rgb = denormalize(img)
            overlay = overlay_segmentation(rgb, pred)
            gt_overlay = overlay_segmentation(rgb, mask)
            Image.fromarray(np.concatenate([rgb, gt_overlay, overlay], axis=1)).save(
                out_dir / f"{name}.png"
            )

    if mious:
        print(f"[prototype] mean mIoU trên {len(mious)} ảnh: {np.mean(mious):.3f}")


def run_cluster(args, extractor, dataset):
    classes = OXFORD_PET_TRIMAP_CLASSES
    images = [dataset[i][0] for i in range(args.num_references)]
    segmenter = ClusteringSegmenter(extractor, num_clusters=len(classes))
    segmenter.fit(images)

    out_dir = Path(args.output_dir) if args.output_dir else None
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)

    start = args.num_references
    stop = start + args.num_samples
    for i in range(start, stop):
        img, _, name = dataset[i]
        pred = segmenter.segment(img).cpu()
        print(f"[cluster] {name}: cluster map shape={tuple(pred.shape)}")
        if out_dir is not None:
            from PIL import Image
            rgb = denormalize(img)
            color = labels_to_color(pred)
            Image.fromarray(np.concatenate([rgb, color], axis=1)).save(out_dir / f"{name}.png")


def main():
    parser = argparse.ArgumentParser(description="Zero-shot segmentation với DINOv2.")
    parser.add_argument("--data_root", type=str, default="../data/oxford_pet")
    parser.add_argument("--method", type=str, default="prototype", choices=["prototype", "cluster"])
    parser.add_argument(
        "--model", type=str, default="dinov2_vits14",
        choices=["dinov2_vits14", "dinov2_vitb14", "dinov2_vitl14", "dinov2_vitg14"],
    )
    parser.add_argument("--image_size", type=int, default=224, help="bội số của 14")
    parser.add_argument("--num_references", type=int, default=3)
    parser.add_argument("--num_samples", type=int, default=5)
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--output_dir", type=str, default=None, help="nếu set → lưu ảnh overlay.")
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA không khả dụng — fallback CPU.")
        args.device = "cpu"

    extractor = DINOv2DenseExtractor(model_name=args.model, device=args.device)
    dataset = OxfordPetSegDataset(
        args.data_root, image_size=args.image_size,
        max_samples=args.num_references + args.num_samples,
    )
    print(f"Loaded {len(dataset)} ảnh từ {args.data_root}; device={args.device}")

    if args.method == "prototype":
        run_prototype(args, extractor, dataset)
    else:
        run_cluster(args, extractor, dataset)


if __name__ == "__main__":
    main()
