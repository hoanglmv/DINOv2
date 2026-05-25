"""Batch evaluation cho zero-shot segmentation trên Oxford-IIIT Pet.

Quy trình (training-free):
    1. Tải dataset (split='test').
    2. Với mỗi backbone (DINOv2-S, DINOv2-B, ViT-B/16-ImageNet) và mỗi method
       (prototype / cluster), tính mIoU + per-class IoU + pixel accuracy.
    3. Dump kết quả ra JSON + lưu một số ảnh đại diện (best/worst) để vẽ figure.

Cách chạy (từ thư mục src/):
    python -m zero_shot_seg.evaluate --data_root ../data/oxford_pet \
        --num_refs 10 --num_eval 150 --device cpu \
        --output_json ../logs/zero_shot_seg_results.json \
        --samples_dir ../logs/seg_samples
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from .classes import OXFORD_PET_TRIMAP_CLASSES
from .data import OxfordPetSegDataset, denormalize
from .feature_extractor import DINOv2DenseExtractor, ImageNetViTDenseExtractor
from .metrics import (
    accumulate_metrics,
    match_clusters_to_classes,
    remap_cluster_labels,
    segmentation_metrics,
)
from .segmenter import ClusteringSegmenter, PrototypeZeroShotSegmenter
from .visualize import overlay_segmentation


BACKBONES = ["dinov2_vits14", "dinov2_vitb14", "vit_b_16_in1k"]
METHODS = ["prototype", "cluster"]


def build_extractor(name: str, device: str):
    if name == "vit_b_16_in1k":
        return ImageNetViTDenseExtractor(device=device)
    return DINOv2DenseExtractor(model_name=name, device=device)


def image_size_for(extractor) -> int:
    """Image size hợp lệ cho extractor: 224 cho cả DINOv2 (14*16) và ViT-IN (16*14)."""
    return 224


def run_prototype(extractor, dataset, num_refs: int, num_eval: int, num_classes: int):
    classes = OXFORD_PET_TRIMAP_CLASSES
    refs = {c: [] for c in classes}
    for i in range(num_refs):
        img, mask, _ = dataset[i]
        for cls_idx, cls in enumerate(classes):
            cls_mask = mask == cls_idx
            if cls_mask.sum() > 0:
                refs[cls].append((img, cls_mask))

    segmenter = PrototypeZeroShotSegmenter(extractor, classes)
    segmenter.build_prototypes(refs)

    preds: List[torch.Tensor] = []
    targets: List[torch.Tensor] = []
    per_sample: List[Tuple[str, float]] = []
    for i in tqdm(range(num_refs, num_refs + num_eval), desc="prototype", leave=False):
        img, mask, name = dataset[i]
        pred = segmenter.segment(img).cpu()
        preds.append(pred)
        targets.append(mask)
        per_sample.append((name, segmentation_metrics(pred, mask, num_classes)["mIoU"]))

    return preds, targets, per_sample


def run_cluster(extractor, dataset, num_refs: int, num_eval: int, num_classes: int):
    classes = OXFORD_PET_TRIMAP_CLASSES
    fit_images = [dataset[i][0] for i in range(num_refs)]
    segmenter = ClusteringSegmenter(extractor, num_clusters=num_classes)
    segmenter.fit(fit_images)

    # Học mapping cluster → class trên chính tập reference (không nhìn vào eval).
    cluster_pred_refs = torch.cat([segmenter.segment(img).cpu().flatten() for img in fit_images])
    target_refs = torch.cat([dataset[i][1].flatten() for i in range(num_refs)])
    mapping = match_clusters_to_classes(
        cluster_pred_refs, target_refs, num_clusters=num_classes, num_classes=num_classes
    )

    preds: List[torch.Tensor] = []
    targets: List[torch.Tensor] = []
    per_sample: List[Tuple[str, float]] = []
    for i in tqdm(range(num_refs, num_refs + num_eval), desc="cluster", leave=False):
        img, mask, name = dataset[i]
        raw = segmenter.segment(img).cpu()
        pred = remap_cluster_labels(raw, mapping)
        preds.append(pred)
        targets.append(mask)
        per_sample.append((name, segmentation_metrics(pred, mask, num_classes)["mIoU"]))

    return preds, targets, per_sample, mapping


def dump_qualitative_samples(
    dataset, results_by_key: Dict[Tuple[str, str], List[torch.Tensor]],
    sample_indices: List[int], samples_dir: Path,
) -> None:
    """Lưu ảnh gốc, ground truth, và prediction (mỗi backbone × method)
    cho một vài ảnh đại diện để plot_results sử dụng.

    Layout:
        samples_dir/
            <name>/original.png
            <name>/gt.png
            <name>/<backbone>_<method>.png  (label map đã uint8, palette áp dụng khi vẽ)
    """
    samples_dir.mkdir(parents=True, exist_ok=True)
    for local_idx, ds_idx in enumerate(sample_indices):
        img, mask, name = dataset[ds_idx]
        out_dir = samples_dir / name
        out_dir.mkdir(parents=True, exist_ok=True)

        rgb = denormalize(img)
        Image.fromarray(rgb).save(out_dir / "original.png")
        Image.fromarray(mask.numpy().astype(np.uint8)).save(out_dir / "gt.png")

        for (backbone, method), preds in results_by_key.items():
            pred = preds[local_idx]
            Image.fromarray(pred.numpy().astype(np.uint8)).save(
                out_dir / f"{backbone}_{method}.png"
            )


def main():
    parser = argparse.ArgumentParser(description="Batch eval zero-shot segmentation.")
    parser.add_argument("--data_root", type=str, default="../data/oxford_pet")
    parser.add_argument("--num_refs", type=int, default=10)
    parser.add_argument("--num_eval", type=int, default=150)
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    parser.add_argument(
        "--backbones", type=str, nargs="+", default=BACKBONES, choices=BACKBONES,
    )
    parser.add_argument(
        "--methods", type=str, nargs="+", default=METHODS, choices=METHODS,
    )
    parser.add_argument("--output_json", type=str, default="../logs/zero_shot_seg_results.json")
    parser.add_argument("--samples_dir", type=str, default="../logs/seg_samples")
    parser.add_argument(
        "--num_samples", type=int, default=4,
        help="Số ảnh đại diện lưu cho phân tích định tính (best + worst).",
    )
    parser.add_argument("--split", type=str, default="test", choices=["trainval", "test"])
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA không khả dụng — fallback CPU.")
        args.device = "cpu"

    num_classes = len(OXFORD_PET_TRIMAP_CLASSES)
    dataset = OxfordPetSegDataset(
        args.data_root, split=args.split, image_size=224,
        max_samples=args.num_refs + args.num_eval,
    )
    print(f"Dataset: {len(dataset)} ảnh từ split='{args.split}' (refs={args.num_refs}, eval={args.num_eval}).")

    all_results: Dict[str, Dict[str, dict]] = {}
    all_preds: Dict[Tuple[str, str], List[torch.Tensor]] = {}
    all_per_sample: Dict[Tuple[str, str], List[Tuple[str, float]]] = {}
    targets_cached: List[torch.Tensor] = []

    for backbone in args.backbones:
        print(f"\n=== {backbone} ===")
        extractor = build_extractor(backbone, args.device)
        all_results[backbone] = {}

        for method in args.methods:
            if method == "prototype":
                preds, targets, per_sample = run_prototype(
                    extractor, dataset, args.num_refs, args.num_eval, num_classes
                )
            else:
                preds, targets, per_sample, _mapping = run_cluster(
                    extractor, dataset, args.num_refs, args.num_eval, num_classes
                )

            if not targets_cached:
                targets_cached = targets

            metrics = accumulate_metrics(
                preds, targets, num_classes, class_names=OXFORD_PET_TRIMAP_CLASSES,
            )
            all_results[backbone][method] = metrics
            all_preds[(backbone, method)] = preds
            all_per_sample[(backbone, method)] = per_sample

            print(
                f"  [{method}] mIoU={metrics['mIoU']:.4f} "
                f"pixel_acc={metrics['pixel_acc']:.4f} "
                f"per_class={ {k: (f'{v:.3f}' if v is not None else 'NaN') for k, v in metrics['per_class_iou'].items()} }"
            )

        # Giải phóng để giảm áp lực RAM trước backbone tiếp theo.
        del extractor

    out_json = Path(args.output_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "config": {
            "num_refs": args.num_refs,
            "num_eval": args.num_eval,
            "split": args.split,
            "device": args.device,
            "backbones": args.backbones,
            "methods": args.methods,
            "class_names": OXFORD_PET_TRIMAP_CLASSES,
        },
        "results": all_results,
    }
    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    print(f"\n→ Đã ghi metrics vào {out_json}")

    # Chọn ảnh đại diện: 2 best + 2 worst theo DINOv2-B prototype (nếu có), fallback backbone đầu tiên.
    ranking_key = ("dinov2_vitb14", "prototype")
    if ranking_key not in all_per_sample:
        ranking_key = next(iter(all_per_sample.keys()))
    ranked = sorted(all_per_sample[ranking_key], key=lambda x: x[1])
    n = min(args.num_samples, len(ranked))
    half = max(1, n // 2)
    bottom = [name for name, _ in ranked[:half]]
    top = [name for name, _ in ranked[-(n - half):]] if n > half else []
    seen = set()
    chosen = []
    for nm in top + bottom:
        if nm not in seen:
            seen.add(nm)
            chosen.append(nm)

    # Map name → local idx (trong eval range).
    name_to_local = {name: i for i, (name, _) in enumerate(all_per_sample[ranking_key])}
    sample_indices_local = [name_to_local[name] for name in chosen]
    sample_indices_ds = [args.num_refs + i for i in sample_indices_local]

    samples_dir = Path(args.samples_dir)
    dump_qualitative_samples(dataset, all_preds, sample_indices_ds, samples_dir)
    # Ghi danh sách tên (theo thứ tự top-first → bottom-last) để plot_results biết thứ tự.
    (samples_dir / "sample_order.json").write_text(json.dumps({
        "ranking_key": list(ranking_key),
        "names": chosen,
        "miou": {name: float(score) for name, score in all_per_sample[ranking_key]},
    }, indent=2, ensure_ascii=False))
    print(f"→ Đã lưu {len(chosen)} ảnh đại diện vào {samples_dir}")


if __name__ == "__main__":
    main()
