"""Trích xuất dense patch features cho bài toán segmentation.

Khác với classification (chỉ dùng CLS token), segmentation cần features ở mức
**patch** để giữ thông tin không gian. DINOv2 expose `forward_features` trả về
dict trong đó `x_norm_patchtokens` có shape (B, N, D) với N = (H/patch) * (W/patch).

Ngoài DINOv2, file này cung cấp thêm baseline `ImageNetViTDenseExtractor` (ViT-B/16
giám sát trên ImageNet) để so sánh chất lượng patch features cho dense task.
"""

import torch
import torch.nn as nn


class DINOv2DenseExtractor(nn.Module):
    """Wrapper trích patch tokens của DINOv2.

    Args:
        model_name: 'dinov2_vits14' (nhẹ, ~21M), 'dinov2_vitb14' (~86M),
            'dinov2_vitl14' (~300M), 'dinov2_vitg14' (~1.1B).
        device: 'cpu' hoặc 'cuda'.
        freeze: backbone luôn ở chế độ eval và không có gradient
            (zero-shot/training-free).
    """

    VALID_MODELS = ("dinov2_vits14", "dinov2_vitb14", "dinov2_vitl14", "dinov2_vitg14")

    def __init__(self, model_name: str = "dinov2_vits14", device: str = "cpu", freeze: bool = True):
        super().__init__()
        if model_name not in self.VALID_MODELS:
            raise ValueError(f"model_name phải thuộc {self.VALID_MODELS}, nhận: {model_name}")
        self.model_name = model_name
        self.device = torch.device(device)

        # Tải DINOv2 từ torch.hub (lần đầu sẽ download weights).
        self.backbone = torch.hub.load("facebookresearch/dinov2", model_name)
        self.backbone.to(self.device)

        if freeze:
            self.backbone.eval()
            for p in self.backbone.parameters():
                p.requires_grad = False

        self.patch_size = self.backbone.patch_size      # 14 với tất cả model DINOv2
        self.embed_dim = self.backbone.embed_dim        # 384 / 768 / 1024 / 1536

    def _check_input(self, x: torch.Tensor) -> None:
        if x.dim() != 4:
            raise ValueError(f"Input phải có shape (B, 3, H, W), nhận shape={tuple(x.shape)}")
        _, c, H, W = x.shape
        if c != 3:
            raise ValueError(f"Số channel phải là 3, nhận {c}")
        if H % self.patch_size != 0 or W % self.patch_size != 0:
            raise ValueError(
                f"H và W phải chia hết cho patch_size={self.patch_size}, "
                f"nhận (H,W)=({H},{W})"
            )

    @torch.no_grad()
    def extract(self, x: torch.Tensor) -> torch.Tensor:
        """Lấy dense patch features.

        Args:
            x: tensor (B, 3, H, W), đã chuẩn hoá theo ImageNet mean/std.
        Returns:
            tensor (B, D, h, w) với h=H/14, w=W/14, D=embed_dim.
        """
        self._check_input(x)
        B, _, H, W = x.shape
        h, w = H // self.patch_size, W // self.patch_size
        x = x.to(self.device)

        feats = self.backbone.forward_features(x)  # type: ignore[attr-defined]
        # `x_norm_patchtokens`: (B, h*w, D) — đầu ra patch token đã LayerNorm.
        patch_tokens = feats["x_norm_patchtokens"]
        # Reshape về không gian 2D: (B, h, w, D) -> (B, D, h, w).
        return patch_tokens.reshape(B, h, w, -1).permute(0, 3, 1, 2).contiguous()


class ImageNetViTDenseExtractor(nn.Module):
    """Baseline: ViT-B/16 huấn luyện có giám sát trên ImageNet, trích patch tokens.

    Cho phép so sánh chất lượng dense feature giữa SSL (DINOv2) và supervised (ViT-IN).
    Giữ cùng interface với DINOv2DenseExtractor: `.extract(x) → (B, D, h, w)`,
    `.device`, `.patch_size`, `.embed_dim`.
    """

    PATCH_SIZE = 16
    EMBED_DIM = 768

    def __init__(self, device: str = "cpu", freeze: bool = True):
        super().__init__()
        from torchvision.models import vit_b_16, ViT_B_16_Weights

        self.device = torch.device(device)
        self.weights = ViT_B_16_Weights.IMAGENET1K_V1
        self.backbone = vit_b_16(weights=self.weights)
        self.backbone.to(self.device)

        if freeze:
            self.backbone.eval()
            for p in self.backbone.parameters():
                p.requires_grad = False

        self.patch_size = self.PATCH_SIZE
        self.embed_dim = self.EMBED_DIM

    def _check_input(self, x: torch.Tensor) -> None:
        if x.dim() != 4:
            raise ValueError(f"Input phải có shape (B, 3, H, W), nhận shape={tuple(x.shape)}")
        _, c, H, W = x.shape
        if c != 3:
            raise ValueError(f"Số channel phải là 3, nhận {c}")
        if H % self.patch_size != 0 or W % self.patch_size != 0:
            raise ValueError(
                f"H và W phải chia hết cho patch_size={self.patch_size}, "
                f"nhận (H,W)=({H},{W})"
            )
        # torchvision ViT-B/16 dùng position embedding cố định cho image_size=224.
        if H != 224 or W != 224:
            raise ValueError(
                "ImageNetViTDenseExtractor yêu cầu input 224x224 (position embedding cố định)."
            )

    @torch.no_grad()
    def extract(self, x: torch.Tensor) -> torch.Tensor:
        """Lấy patch token sequence (đã qua encoder Transformer), bỏ CLS token.

        Quy trình giống `VisionTransformer.forward()` ngoại trừ việc dừng trước head:
            x → _process_input → prepend CLS + pos_embed (qua encoder) → bỏ CLS → reshape.
        """
        self._check_input(x)
        B, _, H, W = x.shape
        h, w = H // self.patch_size, W // self.patch_size
        x = x.to(self.device)

        # Tách patch + nhúng theo cách torchvision ViT.
        z = self.backbone._process_input(x)                                # (B, h*w, D)
        cls = self.backbone.class_token.expand(B, -1, -1)                  # (B, 1, D)
        z = torch.cat([cls, z], dim=1)                                     # (B, 1+h*w, D)
        z = self.backbone.encoder(z)                                       # (B, 1+h*w, D)
        patch_tokens = z[:, 1:, :]                                         # (B, h*w, D), bỏ CLS

        return patch_tokens.reshape(B, h, w, -1).permute(0, 3, 1, 2).contiguous()
