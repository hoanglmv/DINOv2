"""Hai segmenter zero-shot/training-free dựa trên DINOv2 patch features."""

from typing import Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn.functional as F

# Mỗi reference có thể là (image, mask) hoặc chỉ image (mask=None → dùng cả ảnh).
Reference = Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]


def _resize_mask_to_grid(mask: torch.Tensor, grid_hw: Tuple[int, int]) -> torch.Tensor:
    """Down-sample mask về kích thước patch grid (h, w) bằng nearest neighbor."""
    if mask.dim() == 2:
        m = mask.float().unsqueeze(0).unsqueeze(0)
    elif mask.dim() == 3:
        m = mask.float().unsqueeze(0)
    else:
        raise ValueError(f"mask phải có 2 hoặc 3 chiều, nhận shape={tuple(mask.shape)}")
    m = F.interpolate(m, size=grid_hw, mode="nearest")
    return m.squeeze(0).squeeze(0)


class PrototypeZeroShotSegmenter:
    """Zero-shot segmentation theo prototype.

    Quy trình:
        1. Với mỗi class, lấy patch features từ một vài ảnh tham chiếu (có thể kèm
           mask đánh dấu vùng thuộc class). Trung bình hoá → prototype (D,).
        2. Với ảnh mới: trích patch features (D, h, w), normalize, tính cosine sim
           với từng prototype → (num_classes, h, w).
        3. Upsample sim map về (H, W), argmax → segmentation label map.
    """

    def __init__(self, feature_extractor, class_names: Sequence[str]):
        if len(class_names) < 2:
            raise ValueError("Cần ít nhất 2 class.")
        self.feature_extractor = feature_extractor
        self.class_names: List[str] = list(class_names)
        self.num_classes = len(self.class_names)
        self.prototypes: Optional[torch.Tensor] = None  # (num_classes, D)

    @torch.no_grad()
    def build_prototypes(self, references: Dict[str, List[Reference]]) -> None:
        """Tính prototype cho từng class.

        Args:
            references: dict {class_name: [reference, ...]}. Mỗi reference là:
                - tensor (3, H, W) — coi toàn ảnh là đại diện cho class, hoặc
                - tuple (image, mask) — mask shape (H, W) bool/0-1 đánh dấu pixel thuộc class.
        """
        missing = [c for c in self.class_names if c not in references or not references[c]]
        if missing:
            raise ValueError(f"Thiếu reference cho các class: {missing}")

        device = self.feature_extractor.device
        protos = []
        for cls in self.class_names:
            class_vectors = []
            for ref in references[cls]:
                if isinstance(ref, tuple):
                    img, mask = ref
                else:
                    img, mask = ref, None

                img = img.unsqueeze(0).to(device)
                feat = self.feature_extractor.extract(img)[0]  # (D, h, w)
                D, h, w = feat.shape

                if mask is None:
                    pooled = feat.mean(dim=(1, 2))
                else:
                    m = _resize_mask_to_grid(mask.to(device), (h, w))
                    sel = feat[:, m > 0.5]
                    if sel.shape[1] == 0:
                        # Không patch nào nằm trong mask sau khi down-sample → bỏ qua.
                        continue
                    pooled = sel.mean(dim=1)
                class_vectors.append(pooled)

            if not class_vectors:
                raise RuntimeError(
                    f"Không thu được vector reference nào cho class '{cls}'. "
                    f"Mask quá nhỏ so với patch grid?"
                )
            proto = torch.stack(class_vectors).mean(dim=0)
            protos.append(F.normalize(proto, dim=0))

        self.prototypes = torch.stack(protos)  # (num_classes, D)

    @torch.no_grad()
    def segment(
        self, image: torch.Tensor, return_logits: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Segment một ảnh (3, H, W). Trả về label map (H, W) (và sim map nếu cần)."""
        if self.prototypes is None:
            raise RuntimeError("Gọi build_prototypes() trước khi segment.")
        if image.dim() != 3:
            raise ValueError(f"image phải có shape (3, H, W), nhận {tuple(image.shape)}")

        device = self.feature_extractor.device
        H, W = image.shape[1:]
        img = image.unsqueeze(0).to(device)
        feat = self.feature_extractor.extract(img)[0]            # (D, h, w)
        D, h, w = feat.shape
        feat_n = F.normalize(feat, dim=0)                        # cosine

        # (C, D) @ (D, h*w) = (C, h*w)
        sim = self.prototypes @ feat_n.reshape(D, -1)
        sim = sim.reshape(self.num_classes, h, w)

        sim_up = F.interpolate(
            sim.unsqueeze(0), size=(H, W), mode="bilinear", align_corners=False
        )[0]
        labels = sim_up.argmax(dim=0)

        if return_logits:
            return labels, sim_up
        return labels


class ClusteringSegmenter:
    """Unsupervised segmentation bằng KMeans trên patch features.

    Cluster ID không có nghĩa ngữ nghĩa — sau khi cluster, người dùng tự gán
    nhãn (cluster_to_class_name). Phù hợp khi chưa có reference rõ ràng.
    """

    def __init__(self, feature_extractor, num_clusters: int):
        if num_clusters < 2:
            raise ValueError("num_clusters phải >= 2.")
        self.feature_extractor = feature_extractor
        self.num_clusters = num_clusters
        self.centroids: Optional[torch.Tensor] = None  # (K, D), đã normalize.

    @torch.no_grad()
    def fit(self, images: Sequence[torch.Tensor], n_iter: int = 20, seed: int = 0) -> None:
        """Gom patch features từ các ảnh rồi chạy spherical KMeans."""
        device = self.feature_extractor.device
        all_feats = []
        for img in images:
            if img.dim() != 3:
                raise ValueError(f"Mỗi image phải có shape (3, H, W), nhận {tuple(img.shape)}")
            feat = self.feature_extractor.extract(img.unsqueeze(0).to(device))[0]  # (D, h, w)
            D = feat.shape[0]
            all_feats.append(feat.reshape(D, -1).T)  # (h*w, D)

        x = torch.cat(all_feats, dim=0)
        x = F.normalize(x, dim=1)
        self.centroids = self._spherical_kmeans(x, self.num_clusters, n_iter=n_iter, seed=seed)

    @staticmethod
    def _spherical_kmeans(x: torch.Tensor, k: int, n_iter: int = 20, seed: int = 0) -> torch.Tensor:
        """KMeans với cosine similarity (centroid được L2-normalize sau mỗi vòng)."""
        N, D = x.shape
        if N < k:
            raise ValueError(f"Số sample ({N}) < số cluster ({k}).")
        g = torch.Generator(device=x.device).manual_seed(seed)
        idx = torch.randperm(N, generator=g, device=x.device)[:k]
        c = x[idx].clone()

        for _ in range(n_iter):
            assign = (x @ c.T).argmax(dim=1)              # (N,)
            new_c = torch.zeros_like(c)
            for j in range(k):
                m = assign == j
                if m.any():
                    new_c[j] = x[m].mean(dim=0)
                else:
                    new_c[j] = c[j]                       # cluster rỗng → giữ nguyên
            new_c = F.normalize(new_c, dim=1)
            shift = (new_c - c).abs().sum().item()
            c = new_c
            if shift < 1e-4:
                break
        return c

    @torch.no_grad()
    def segment(self, image: torch.Tensor) -> torch.Tensor:
        """Trả về cluster-id map (H, W)."""
        if self.centroids is None:
            raise RuntimeError("Gọi fit() trước khi segment.")
        if image.dim() != 3:
            raise ValueError(f"image phải có shape (3, H, W), nhận {tuple(image.shape)}")
        device = self.feature_extractor.device
        H, W = image.shape[1:]
        feat = self.feature_extractor.extract(image.unsqueeze(0).to(device))[0]  # (D, h, w)
        D, h, w = feat.shape
        feat_n = F.normalize(feat.reshape(D, -1).T, dim=1)                       # (h*w, D)
        labels_small = (feat_n @ self.centroids.T).argmax(dim=1).reshape(h, w)
        labels = F.interpolate(
            labels_small.float().unsqueeze(0).unsqueeze(0),
            size=(H, W), mode="nearest",
        )[0, 0].long()
        return labels
