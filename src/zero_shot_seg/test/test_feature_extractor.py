"""Kiểm tra interface của DINOv2DenseExtractor mà không cần tải weights.

Chạy:
    cd src
    python -m zero_shot_seg.test.test_feature_extractor

Có hai chế độ:
  - Mặc định: kiểm tra validation input bằng FakeDenseExtractor (nhanh, không cần mạng).
  - Khi đặt env RUN_REAL_DINO=1: tải thật DINOv2 ViT-S/14 và chạy 1 forward (~vài chục giây CPU).
"""

import os

import torch

from zero_shot_seg.test._fake_extractor import FakeDenseExtractor


def test_fake_extractor_shape():
    fx = FakeDenseExtractor(embed_dim=32, patch_size=14)
    x = torch.randn(2, 3, 28, 42)  # 28=14*2, 42=14*3
    out = fx.extract(x)
    assert out.shape == (2, 32, 2, 3), f"shape sai: {out.shape}"
    print("ok: FakeDenseExtractor → shape (B, D, h, w) đúng.")


def test_fake_extractor_rejects_bad_size():
    fx = FakeDenseExtractor(embed_dim=8, patch_size=14)
    try:
        fx.extract(torch.randn(1, 3, 30, 28))  # 30 không chia hết 14
    except ValueError:
        print("ok: từ chối input không chia hết cho patch_size.")
        return
    raise AssertionError("Đáng lẽ phải raise ValueError.")


def test_fake_extractor_rejects_bad_channels():
    fx = FakeDenseExtractor(embed_dim=8, patch_size=14)
    try:
        fx.extract(torch.randn(1, 1, 28, 28))
    except ValueError:
        print("ok: từ chối input không có 3 channel.")
        return
    raise AssertionError("Đáng lẽ phải raise ValueError.")


def test_real_dinov2_if_requested():
    if os.environ.get("RUN_REAL_DINO") != "1":
        print("bỏ qua test thật DINOv2 (set RUN_REAL_DINO=1 để bật).")
        return
    from zero_shot_seg.feature_extractor import DINOv2DenseExtractor
    fx = DINOv2DenseExtractor(model_name="dinov2_vits14", device="cpu")
    x = torch.randn(1, 3, 28, 28)  # 2x2 patch — input cực nhỏ để chạy nhanh.
    out = fx.extract(x)
    assert out.shape == (1, fx.embed_dim, 2, 2), f"shape sai: {out.shape}"
    print(f"ok: DINOv2 thật trả về shape {tuple(out.shape)}.")


if __name__ == "__main__":
    test_fake_extractor_shape()
    test_fake_extractor_rejects_bad_size()
    test_fake_extractor_rejects_bad_channels()
    test_real_dinov2_if_requested()
    print("\nAll feature_extractor tests passed.")
