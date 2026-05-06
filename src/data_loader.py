import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os

# Biến toàn cục định nghĩa phép biến đổi ảnh chuẩn cho DINOv2
DINO_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                         std=[0.229, 0.224, 0.225]),
])

def get_cifar10_dataloaders(batch_size=32, num_workers=2):
    """
    Tải dataset CIFAR-10 (Tự động tải từ mạng nếu chưa có).
    """
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
    os.makedirs(data_dir, exist_ok=True)

    print("Loading CIFAR-10 training data...")
    train_dataset = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=DINO_TRANSFORM)
    
    print("Loading CIFAR-10 test data...")
    test_dataset = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=DINO_TRANSFORM)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, test_loader, train_dataset.classes


def get_intel_image_dataloaders(batch_size=32, num_workers=2):
    """
    Tải dataset Intel Image Classification từ thư mục cục bộ.
    Yêu cầu cấu trúc: dataset/IntelImageClassification/seg_train và seg_test
    """
    base_dir = os.path.dirname(os.path.dirname(__file__))
    train_dir = os.path.join(base_dir, 'dataset', 'IntelImageClassification', 'seg_train')
    test_dir = os.path.join(base_dir, 'dataset', 'IntelImageClassification', 'seg_test')

    print("Loading Intel Image Classification training data...")
    train_dataset = datasets.ImageFolder(root=train_dir, transform=DINO_TRANSFORM)
    
    print("Loading Intel Image Classification test data...")
    test_dataset = datasets.ImageFolder(root=test_dir, transform=DINO_TRANSFORM)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, test_loader, train_dataset.classes


def get_custom_kaggle_dataloaders(dataset_folder_name, train_folder='train', test_folder='val', batch_size=32, num_workers=2):
    """
    Hàm tổng quát (Template) để tải các dataset Kaggle khác có cấu trúc thư mục con là các class.
    Ví dụ: get_custom_kaggle_dataloaders('CassavaLeafDisease', 'train', 'test')
    """
    base_dir = os.path.dirname(os.path.dirname(__file__))
    train_dir = os.path.join(base_dir, 'dataset', dataset_folder_name, train_folder)
    test_dir = os.path.join(base_dir, 'dataset', dataset_folder_name, test_folder)

    print(f"Loading {dataset_folder_name} training data...")
    train_dataset = datasets.ImageFolder(root=train_dir, transform=DINO_TRANSFORM)
    
    print(f"Loading {dataset_folder_name} test data...")
    test_dataset = datasets.ImageFolder(root=test_dir, transform=DINO_TRANSFORM)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, test_loader, train_dataset.classes
