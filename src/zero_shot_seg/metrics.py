"""Metric cho semantic segmentation: per-class IoU, mIoU, pixel accuracy.

Dùng tensor thuần (không phụ thuộc sklearn) để giữ pipeline nhẹ.
"""

from typing import Dict, List, Optional

import numpy as np
import torch


def confusion_matrix(
    pred: torch.Tensor, target: torch.Tensor, num_classes: int
) -> torch.Tensor:
    """Confusion matrix (num_classes, num_classes) với hàng = ground truth, cột = dự đoán."""
    if pred.shape != target.shape:
        raise ValueError(f"pred {tuple(pred.shape)} khác shape target {tuple(target.shape)}")
    p = pred.flatten().long()
    t = target.flatten().long()
    mask = (t >= 0) & (t < num_classes) & (p >= 0) & (p < num_classes)
    idx = t[mask] * num_classes + p[mask]
    bins = torch.bincount(idx, minlength=num_classes * num_classes)
    return bins.reshape(num_classes, num_classes)


def per_class_iou_from_cm(cm: torch.Tensor) -> torch.Tensor:
    """IoU mỗi class (NaN khi không có pixel nào của class đó trong batch)."""
    cm = cm.float()
    inter = torch.diag(cm)
    union = cm.sum(dim=1) + cm.sum(dim=0) - inter
    iou = torch.where(union > 0, inter / union, torch.full_like(union, float("nan")))
    return iou


def segmentation_metrics(
    pred: torch.Tensor, target: torch.Tensor, num_classes: int
) -> Dict[str, float]:
    """Metric cho MỘT ảnh: trả về dict `{mIoU, pixel_acc, iou_<c>}`."""
    cm = confusion_matrix(pred, target, num_classes)
    iou = per_class_iou_from_cm(cm)
    pixel_acc = (torch.diag(cm).sum().float() / cm.sum().clamp_min(1)).item()
    out = {"mIoU": float(torch.nanmean(iou).item()), "pixel_acc": pixel_acc}
    for c in range(num_classes):
        out[f"iou_{c}"] = float(iou[c].item())  # có thể là NaN
    return out


def accumulate_metrics(
    preds: List[torch.Tensor],
    targets: List[torch.Tensor],
    num_classes: int,
    class_names: Optional[List[str]] = None,
) -> Dict[str, object]:
    """Cộng dồn confusion matrix qua nhiều ảnh để có metric dataset-level.

    Trả về:
        - mIoU: trung bình per-class IoU (NaN class bị bỏ qua).
        - pixel_acc: tổng số pixel đúng / tổng pixel.
        - per_class_iou: dict {name: iou}.
    """
    if len(preds) != len(targets):
        raise ValueError(f"len(preds)={len(preds)} != len(targets)={len(targets)}")
    if class_names is None:
        class_names = [str(i) for i in range(num_classes)]
    if len(class_names) != num_classes:
        raise ValueError(f"class_names có {len(class_names)} phần tử, cần {num_classes}")

    cm_total = torch.zeros(num_classes, num_classes, dtype=torch.long)
    for p, t in zip(preds, targets):
        cm_total = cm_total + confusion_matrix(p, t, num_classes)

    iou = per_class_iou_from_cm(cm_total)
    miou = float(torch.nanmean(iou).item())
    pixel_acc = float((torch.diag(cm_total).sum().float() / cm_total.sum().clamp_min(1)).item())

    per_class = {}
    for c, name in enumerate(class_names):
        v = iou[c].item()
        per_class[name] = None if np.isnan(v) else float(v)

    return {
        "mIoU": miou,
        "pixel_acc": pixel_acc,
        "per_class_iou": per_class,
    }


def match_clusters_to_classes(
    cluster_pred: torch.Tensor, target: torch.Tensor, num_clusters: int, num_classes: int
) -> Dict[int, int]:
    """Gán mỗi cluster id → class id bằng greedy IoU matching.

    Với mỗi cluster, chọn class có IoU cao nhất khi xét toàn bộ pixel của cluster đó.
    Cho phép nhiều cluster cùng map về 1 class (đặc biệt khi num_clusters > num_classes).
    """
    if cluster_pred.shape != target.shape:
        raise ValueError(f"cluster_pred {tuple(cluster_pred.shape)} khác target {tuple(target.shape)}")
    p = cluster_pred.flatten().long()
    t = target.flatten().long()

    mapping: Dict[int, int] = {}
    for k in range(num_clusters):
        m = p == k
        if not m.any():
            mapping[k] = 0
            continue
        best_iou = -1.0
        best_c = 0
        for c in range(num_classes):
            tc = t == c
            inter = (m & tc).sum().item()
            union = (m | tc).sum().item()
            if union == 0:
                continue
            iou = inter / union
            if iou > best_iou:
                best_iou = iou
                best_c = c
        mapping[k] = best_c
    return mapping


def remap_cluster_labels(cluster_pred: torch.Tensor, mapping: Dict[int, int]) -> torch.Tensor:
    """Áp dụng mapping cluster_id → class_id lên label map."""
    out = torch.zeros_like(cluster_pred)
    for k, c in mapping.items():
        out[cluster_pred == k] = c
    return out
