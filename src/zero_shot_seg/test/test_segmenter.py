"""Sanity check cho PrototypeZeroShotSegmenter và ClusteringSegmenter.

Dùng FakeDenseExtractor để không cần download DINOv2. Mục tiêu chính là
verify shape, hợp lệ của API, và xử lý lỗi.

Chạy:
    cd src
    python -m zero_shot_seg.test.test_segmenter
"""

import torch

from zero_shot_seg.segmenter import ClusteringSegmenter, PrototypeZeroShotSegmenter
from zero_shot_seg.test._fake_extractor import FakeDenseExtractor


CLASSES = ["fg", "bg"]
IMG_SIZE = 56  # = 14*4 → patch grid 4x4


def _make_reference(class_idx: int) -> tuple:
    """Ảnh nửa trên thuộc class 0, nửa dưới thuộc class 1 (mask binary)."""
    img = torch.randn(3, IMG_SIZE, IMG_SIZE)
    mask = torch.zeros(IMG_SIZE, IMG_SIZE)
    if class_idx == 0:
        mask[: IMG_SIZE // 2] = 1.0
    else:
        mask[IMG_SIZE // 2 :] = 1.0
    return img, mask


def test_prototype_basic_flow():
    fx = FakeDenseExtractor(embed_dim=16, patch_size=14)
    seg = PrototypeZeroShotSegmenter(fx, CLASSES)
    refs = {
        "fg": [_make_reference(0) for _ in range(2)],
        "bg": [_make_reference(1) for _ in range(2)],
    }
    seg.build_prototypes(refs)
    assert seg.prototypes is not None and seg.prototypes.shape == (2, 16)

    test_img = torch.randn(3, IMG_SIZE, IMG_SIZE)
    labels = seg.segment(test_img)
    assert labels.shape == (IMG_SIZE, IMG_SIZE), labels.shape
    assert labels.min().item() >= 0 and labels.max().item() < len(CLASSES)
    print(f"ok: PrototypeZeroShotSegmenter → labels shape {tuple(labels.shape)}")

    labels2, sim = seg.segment(test_img, return_logits=True)
    assert sim.shape == (len(CLASSES), IMG_SIZE, IMG_SIZE), sim.shape
    assert torch.equal(labels, labels2)
    print(f"ok: return_logits → sim shape {tuple(sim.shape)}")


def test_prototype_missing_reference_raises():
    fx = FakeDenseExtractor(embed_dim=8, patch_size=14)
    seg = PrototypeZeroShotSegmenter(fx, CLASSES)
    try:
        seg.build_prototypes({"fg": [_make_reference(0)]})  # thiếu 'bg'
    except ValueError:
        print("ok: thiếu reference → raise ValueError.")
        return
    raise AssertionError("Phải raise ValueError khi thiếu class.")


def test_prototype_segment_before_build_raises():
    fx = FakeDenseExtractor(embed_dim=8, patch_size=14)
    seg = PrototypeZeroShotSegmenter(fx, CLASSES)
    try:
        seg.segment(torch.randn(3, IMG_SIZE, IMG_SIZE))
    except RuntimeError:
        print("ok: segment() trước build_prototypes() → raise RuntimeError.")
        return
    raise AssertionError("Phải raise RuntimeError.")


def test_prototype_with_image_only_reference():
    """Reference không kèm mask → dùng cả ảnh."""
    fx = FakeDenseExtractor(embed_dim=8, patch_size=14)
    seg = PrototypeZeroShotSegmenter(fx, CLASSES)
    refs = {
        "fg": [torch.randn(3, IMG_SIZE, IMG_SIZE)],
        "bg": [torch.randn(3, IMG_SIZE, IMG_SIZE)],
    }
    seg.build_prototypes(refs)
    labels = seg.segment(torch.randn(3, IMG_SIZE, IMG_SIZE))
    assert labels.shape == (IMG_SIZE, IMG_SIZE)
    print("ok: reference dạng image-only (không mask) hoạt động.")


def test_cluster_basic_flow():
    fx = FakeDenseExtractor(embed_dim=12, patch_size=14)
    seg = ClusteringSegmenter(fx, num_clusters=3)
    images = [torch.randn(3, IMG_SIZE, IMG_SIZE) for _ in range(2)]
    seg.fit(images, n_iter=5)
    assert seg.centroids is not None and seg.centroids.shape == (3, 12)
    labels = seg.segment(torch.randn(3, IMG_SIZE, IMG_SIZE))
    assert labels.shape == (IMG_SIZE, IMG_SIZE)
    assert labels.min().item() >= 0 and labels.max().item() < 3
    print(f"ok: ClusteringSegmenter → labels shape {tuple(labels.shape)}")


def test_cluster_segment_before_fit_raises():
    fx = FakeDenseExtractor(embed_dim=8, patch_size=14)
    seg = ClusteringSegmenter(fx, num_clusters=2)
    try:
        seg.segment(torch.randn(3, IMG_SIZE, IMG_SIZE))
    except RuntimeError:
        print("ok: segment() trước fit() → raise RuntimeError.")
        return
    raise AssertionError("Phải raise RuntimeError.")


if __name__ == "__main__":
    test_prototype_basic_flow()
    test_prototype_missing_reference_raises()
    test_prototype_segment_before_build_raises()
    test_prototype_with_image_only_reference()
    test_cluster_basic_flow()
    test_cluster_segment_before_fit_raises()
    print("\nAll segmenter tests passed.")
