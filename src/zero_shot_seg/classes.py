"""Khai báo các class cho zero-shot segmentation.

Tên class chỉ mang ý nghĩa hiển thị/phân loại — bản thân DINOv2 không có
text-image alignment. Việc gán nhãn được thực hiện qua:
  - prototype tính từ reference images, hoặc
  - cluster ID (gán tên thủ công sau khi clustering).
"""

# Oxford-IIIT Pet trimap: 3 class theo annotation gốc (1=pet, 2=bg, 3=border).
# Index trong list này đã được remap về 0,1,2 trong data.TRIMAP_TO_CLASS_IDX.
OXFORD_PET_TRIMAP_CLASSES = [
    "pet",          # foreground (con vật)
    "background",   # nền
    "border",       # vùng ranh giới / không xác định
]

# Tập class nhị phân — phục vụ các bài toán foreground/background đơn giản.
BINARY_CLASSES = [
    "background",
    "foreground",
]

# Pascal VOC 2012 (21 class, kể cả background) — để mở rộng trong tương lai.
VOC_CLASSES = [
    "background", "aeroplane", "bicycle", "bird", "boat",
    "bottle", "bus", "car", "cat", "chair",
    "cow", "diningtable", "dog", "horse", "motorbike",
    "person", "pottedplant", "sheep", "sofa", "train",
    "tvmonitor",
]
