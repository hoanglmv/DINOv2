"""Mock feature extractor cho test trên CPU (không cần tải DINOv2).

Giả lập đúng interface của DINOv2DenseExtractor: có .extract(x) → (B, D, h, w),
có .device và .patch_size. Bên trong dùng một Conv2d stride=patch_size để
ánh xạ ảnh (B, 3, H, W) → (B, D, H/patch, W/patch).
"""

import torch
import torch.nn as nn


class FakeDenseExtractor(nn.Module):
    def __init__(self, embed_dim: int = 32, patch_size: int = 14, device: str = "cpu", seed: int = 0):
        super().__init__()
        torch.manual_seed(seed)
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.device = torch.device(device)
        self.proj = nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size, bias=False)
        for p in self.proj.parameters():
            p.requires_grad = False
        self.to(self.device)

    @torch.no_grad()
    def extract(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 4 or x.shape[1] != 3:
            raise ValueError(f"Cần shape (B, 3, H, W), nhận {tuple(x.shape)}")
        H, W = x.shape[-2:]
        if H % self.patch_size or W % self.patch_size:
            raise ValueError("H, W phải chia hết cho patch_size.")
        return self.proj(x.to(self.device))
