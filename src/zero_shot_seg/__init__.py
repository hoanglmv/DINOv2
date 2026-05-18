"""Zero-shot segmentation với DINOv2 (training-free).

Hai phương pháp được triển khai:
  - PrototypeZeroShotSegmenter: dùng vài ảnh tham chiếu (reference) cho mỗi class,
    tính prototype patch-feature, rồi gán nhãn theo cosine similarity.
  - ClusteringSegmenter: KMeans trên patch features (không cần nhãn), cluster
    sau đó có thể được gán tên class thủ công.
"""

from .feature_extractor import DINOv2DenseExtractor
from .segmenter import PrototypeZeroShotSegmenter, ClusteringSegmenter
from .classes import (
    OXFORD_PET_TRIMAP_CLASSES,
    BINARY_CLASSES,
    VOC_CLASSES,
)

__all__ = [
    "DINOv2DenseExtractor",
    "PrototypeZeroShotSegmenter",
    "ClusteringSegmenter",
    "OXFORD_PET_TRIMAP_CLASSES",
    "BINARY_CLASSES",
    "VOC_CLASSES",
]
