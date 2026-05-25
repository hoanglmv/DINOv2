"""Unit test cho metrics.py — không cần backbone, chạy hoàn toàn trên tensor giả lập.

Chạy:
    cd src
    python -m zero_shot_seg.test.test_metrics
"""

import math

import torch

from zero_shot_seg.metrics import (
    accumulate_metrics,
    confusion_matrix,
    match_clusters_to_classes,
    per_class_iou_from_cm,
    remap_cluster_labels,
    segmentation_metrics,
)


def test_confusion_matrix_perfect():
    t = torch.tensor([[0, 0], [1, 2]])
    cm = confusion_matrix(t.clone(), t, num_classes=3)
    assert cm.shape == (3, 3)
    # đường chéo = số pixel mỗi class.
    assert cm[0, 0] == 2 and cm[1, 1] == 1 and cm[2, 2] == 1
    assert cm.sum().item() == 4
    print("ok: confusion_matrix khớp khi pred==target.")


def test_per_class_iou_handles_missing_class():
    # class 2 không xuất hiện ở cả pred và target → IoU NaN.
    t = torch.tensor([[0, 0], [1, 1]])
    p = torch.tensor([[0, 1], [1, 1]])
    cm = confusion_matrix(p, t, num_classes=3)
    iou = per_class_iou_from_cm(cm)
    assert math.isnan(iou[2].item()), iou
    # class 0: inter=1, union=2 → 0.5
    assert abs(iou[0].item() - 0.5) < 1e-6, iou[0]
    # class 1: inter=2, union=3 → 2/3
    assert abs(iou[1].item() - 2 / 3) < 1e-6, iou[1]
    print("ok: per_class_iou bỏ qua class trống bằng NaN.")


def test_segmentation_metrics_keys_and_pixel_acc():
    t = torch.tensor([[0, 1], [1, 0]])
    p = torch.tensor([[0, 1], [0, 0]])  # đúng 3/4 pixel
    out = segmentation_metrics(p, t, num_classes=2)
    assert set(["mIoU", "pixel_acc", "iou_0", "iou_1"]).issubset(out.keys())
    assert abs(out["pixel_acc"] - 0.75) < 1e-6
    print(f"ok: segmentation_metrics → {out}")


def test_accumulate_metrics_aggregation():
    targets = [torch.tensor([[0, 0], [1, 1]]), torch.tensor([[0, 1], [1, 1]])]
    preds = [torch.tensor([[0, 0], [1, 0]]), torch.tensor([[0, 1], [1, 1]])]
    out = accumulate_metrics(preds, targets, num_classes=2, class_names=["bg", "fg"])
    assert "mIoU" in out and "pixel_acc" in out and "per_class_iou" in out
    assert set(out["per_class_iou"].keys()) == {"bg", "fg"}
    # Tổng 8 pixel; đúng 7 → pixel_acc = 7/8.
    assert abs(out["pixel_acc"] - 7 / 8) < 1e-6, out
    print(f"ok: accumulate_metrics → {out}")


def test_match_clusters_to_classes_greedy():
    # cluster 0 trùng class 1 hoàn toàn, cluster 1 trùng class 0.
    cluster_pred = torch.tensor([[1, 1], [0, 0]])
    target = torch.tensor([[0, 0], [1, 1]])
    mapping = match_clusters_to_classes(cluster_pred, target, num_clusters=2, num_classes=2)
    assert mapping[0] == 1 and mapping[1] == 0, mapping
    print(f"ok: match_clusters_to_classes → {mapping}")


def test_remap_cluster_labels_roundtrip():
    cluster_pred = torch.tensor([[0, 1], [1, 0]])
    mapping = {0: 5, 1: 9}
    out = remap_cluster_labels(cluster_pred, mapping)
    assert torch.equal(out, torch.tensor([[5, 9], [9, 5]]))
    print("ok: remap_cluster_labels giữ shape, đổi đúng label.")


def test_shape_mismatch_raises():
    try:
        confusion_matrix(torch.zeros(4, 4), torch.zeros(3, 3), num_classes=2)
    except ValueError:
        print("ok: confusion_matrix báo lỗi khi pred/target khác shape.")
        return
    raise AssertionError("Phải raise ValueError.")


if __name__ == "__main__":
    test_confusion_matrix_perfect()
    test_per_class_iou_handles_missing_class()
    test_segmentation_metrics_keys_and_pixel_acc()
    test_accumulate_metrics_aggregation()
    test_match_clusters_to_classes_greedy()
    test_remap_cluster_labels_roundtrip()
    test_shape_mismatch_raises()
    print("\nAll metrics tests passed.")
