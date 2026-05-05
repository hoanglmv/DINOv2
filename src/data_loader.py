import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os

def get_dataloaders(batch_size=32, num_workers=2):
    """
    Tải dataset CIFAR-10 và chuẩn bị DataLoader.
    CIFAR-10 được chọn vì kích thước nhỏ, phù hợp chạy nhanh nghiệm thu dự án.
    """
    # DINOv2 yêu cầu ảnh đầu vào chuẩn hóa theo ImageNet
    # Resize lên 224x224 (kích thước patch size mặc định của ViT)
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225]),
    ])

    # Tạo thư mục data nếu chưa có
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
    os.makedirs(data_dir, exist_ok=True)

    print("Downloading/Loading CIFAR-10 training data...")
    train_dataset = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform)
    
    print("Downloading/Loading CIFAR-10 test data...")
    test_dataset = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, test_loader, train_dataset.classes
