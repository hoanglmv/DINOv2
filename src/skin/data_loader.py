import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split

# Ánh xạ 7 loại bệnh của bộ dữ liệu HAM10000 thành các số nguyên (index) để model dễ học
DX_CLASSES = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
CLASS_TO_IDX = {cls: idx for idx, cls in enumerate(DX_CLASSES)}
IDX_TO_CLASS = {idx: cls for cls, idx in CLASS_TO_IDX.items()}

class SkinCancerDataset(Dataset):
    """
    Lớp Dataset tùy chỉnh cho bộ dữ liệu SkinCancerMNIST (HAM10000).
    """
    def __init__(self, df, img_dirs, transform=None):
        """
        Khởi tạo Dataset.
        Args:
            df (pd.DataFrame): Bảng dữ liệu pandas chứa cột 'image_id' và 'dx' (nhãn bệnh).
            img_dirs (list): Danh sách các đường dẫn chứa thư mục ảnh (vì HAM10000 chia ảnh ra 2 phần).
            transform (callable, optional): Các phép biến đổi hình ảnh (ví dụ: resize, crop, normalize).
        """
        self.df = df
        self.img_dirs = img_dirs
        self.transform = transform

    def __len__(self):
        # Trả về tổng số lượng ảnh trong tập dữ liệu
        return len(self.df)

    def _get_image_path(self, image_id):
        # Hàm hỗ trợ tìm kiếm ảnh xem nó nằm ở thư mục part_1 hay part_2
        img_name = f"{image_id}.jpg"
        for d in self.img_dirs:
            p = os.path.join(d, img_name)
            if os.path.exists(p):
                return p
        raise FileNotFoundError(f"Không tìm thấy ảnh {img_name} trong các thư mục được cung cấp.")

    def __getitem__(self, idx):
        # Lấy ra 1 sample (ảnh và nhãn) tại vị trí idx
        row = self.df.iloc[idx]
        image_id = row['image_id']
        label = CLASS_TO_IDX[row['dx']] # Chuyển đổi nhãn chuỗi thành số nguyên
        
        # Đọc ảnh và chuyển thành định dạng RGB
        img_path = self._get_image_path(image_id)
        image = Image.open(img_path).convert("RGB")
        
        # Áp dụng các phép biến đổi (nếu có)
        if self.transform:
            image = self.transform(image)
            
        return image, label


def get_transforms(model_type="dinov2", is_train=True):
    """
    Tạo các phép biến đổi (transforms) hình ảnh phù hợp với từng loại mô hình.
    Việc chuẩn hóa dữ liệu phải giống hệt lúc mô hình được Pre-train.
    """
    if model_type == "dinov2" or model_type == "resnet50":
        # DINOv2 và ResNet50 đều sử dụng chuẩn hóa của ImageNet
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        target_size = 224
        
        if is_train:
            # Data augmentation mạnh hơn cho tập Train để tránh overfitting
            return transforms.Compose([
                transforms.RandomResizedCrop(target_size, scale=(0.7, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation(90),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)
            ])
        else:
            # Biến đổi cơ bản cho tập Validation / Test (không làm sai lệch dữ liệu)
            return transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(target_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)
            ])
    elif model_type == "clip":
        import open_clip
        # OpenCLIP cung cấp sẵn transform đi kèm với trọng số, ta chỉ việc lấy ra dùng
        _, train_transform, val_transform = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
        return train_transform if is_train else val_transform
    else:
        raise ValueError(f"Không hỗ trợ mô hình: {model_type}")

def get_dataloaders(csv_path, img_dirs, model_type="dinov2", batch_size=32, num_workers=4, test_size=0.2, val_size=0.1):
    """
    Hàm đọc file CSV và chia dữ liệu thành 3 tập: Train, Val, Test.
    Sau đó đóng gói thành các đối tượng DataLoader để Pytorch có thể đọc theo từng Batch.
    """
    # Đọc file CSV metadata
    df = pd.read_csv(csv_path)
    
    # Bước 1: Tách tập Test (20%) ra khỏi toàn bộ dữ liệu. Giữ lại Train+Val (80%)
    # stratify=df['dx'] giúp phân bố các nhãn bệnh được giữ nguyên tỷ lệ
    train_val_df, test_df = train_test_split(df, test_size=test_size, stratify=df['dx'], random_state=42)
    
    # Bước 2: Tách tập Train+Val ra thành tập Train và tập Val.
    # Tỷ lệ val_size là tỷ lệ so với toàn bộ gốc (VD 10%), nên cần quy đổi lại so với train_val_df
    rel_val_size = val_size / (1 - test_size)
    train_df, val_df = train_test_split(train_val_df, test_size=rel_val_size, stratify=train_val_df['dx'], random_state=42)
    
    # Đánh lại chỉ số index để Dataset hoạt động chính xác
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)
    
    # Lấy ra các transform tương ứng với mô hình
    train_transform = get_transforms(model_type, is_train=True)
    val_test_transform = get_transforms(model_type, is_train=False)
    
    # Tạo các objects Dataset
    train_dataset = SkinCancerDataset(train_df, img_dirs, transform=train_transform)
    val_dataset = SkinCancerDataset(val_df, img_dirs, transform=val_test_transform)
    test_dataset = SkinCancerDataset(test_df, img_dirs, transform=val_test_transform)
    
    # Tạo các DataLoaders để sinh ra các Batch trong quá trình huấn luyện
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return train_loader, val_loader, test_loader, len(DX_CLASSES)

if __name__ == "__main__":
    # Đoạn code kiểm tra nhanh xem DataLoader có hoạt động bình thường không
    base_dir = "data/SkinCancer"
    csv = os.path.join(base_dir, "HAM10000_metadata.csv")
    img_dirs = [os.path.join(base_dir, "HAM10000_images_part_1"), os.path.join(base_dir, "HAM10000_images_part_2")]
    
    tr, vl, te, n_classes = get_dataloaders(csv, img_dirs, model_type="dinov2", batch_size=8, num_workers=2)
    print(f"Số lượng batches tập Train: {len(tr)}, Val: {len(vl)}, Test: {len(te)}")
    
    images, labels = next(iter(tr))
    print("Kích thước 1 batch ảnh:", images.shape)
    print("Nhãn của 1 batch:", labels.shape)
