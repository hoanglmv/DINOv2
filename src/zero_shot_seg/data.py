"""Loader cho Oxford-IIIT Pet (trimap segmentation) — dataset nhẹ để demo."""

from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

# Mã trimap theo README của Oxford-IIIT Pet:
#   1 = foreground (pet), 2 = background, 3 = border/uncertain.
# Remap về 0..2 cho thuận với index trong classes.OXFORD_PET_TRIMAP_CLASSES.
TRIMAP_TO_CLASS_IDX = {1: 0, 2: 1, 3: 2}

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def make_image_transform(image_size: int = 224) -> transforms.Compose:
    """Transform cho DINOv2: resize → tensor → normalize ImageNet.

    `image_size` phải là bội số của 14 (patch size của DINOv2). 224 = 14*16.
    """
    if image_size % 14 != 0:
        raise ValueError(f"image_size phải chia hết cho 14, nhận {image_size}.")
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def load_pet_trimap(trimap_path: Path, image_size: int = 224) -> torch.Tensor:
    """Đọc trimap PNG → mask int64 (H, W) với giá trị thuộc {0,1,2}."""
    trimap = Image.open(trimap_path).resize((image_size, image_size), Image.NEAREST)
    arr = np.array(trimap)
    out = np.zeros_like(arr, dtype=np.int64)
    for k, v in TRIMAP_TO_CLASS_IDX.items():
        out[arr == k] = v
    return torch.from_numpy(out)


class OxfordPetSegDataset(Dataset):
    """Oxford-IIIT Pet với trimap mask, output (image, mask, name).

    Layout dữ liệu sau khi chạy scripts/download_oxford_pet.sh:
        <root>/images/<name>.jpg
        <root>/annotations/trimaps/<name>.png
        <root>/annotations/{trainval,test}.txt
    """

    def __init__(
        self,
        root: str,
        split: str = "trainval",
        image_size: int = 224,
        max_samples: Optional[int] = None,
    ):
        self.root = Path(root)
        self.image_dir = self.root / "images"
        self.trimap_dir = self.root / "annotations" / "trimaps"
        list_file = self.root / "annotations" / f"{split}.txt"

        if not list_file.exists():
            raise FileNotFoundError(
                f"Không tìm thấy {list_file}. Hãy chạy scripts/download_oxford_pet.sh trước."
            )

        with open(list_file, "r") as f:
            entries: List[str] = []
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                entries.append(line.split()[0])

        if max_samples is not None:
            entries = entries[:max_samples]

        self.entries = entries
        self.image_size = image_size
        self.image_transform = make_image_transform(image_size)

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        name = self.entries[idx]
        img = Image.open(self.image_dir / f"{name}.jpg").convert("RGB")
        img_t = self.image_transform(img)
        mask = load_pet_trimap(self.trimap_dir / f"{name}.png", self.image_size)
        return img_t, mask, name


def denormalize(image_chw: torch.Tensor) -> np.ndarray:
    """Đảo chuẩn hoá ImageNet → numpy (H, W, 3) uint8 để hiển thị."""
    arr = image_chw.detach().cpu().numpy().transpose(1, 2, 0)
    mean = np.array(IMAGENET_MEAN)
    std = np.array(IMAGENET_STD)
    arr = arr * std + mean
    return np.clip(arr * 255, 0, 255).astype(np.uint8)
