import torch
import torch.nn as nn
from tqdm import tqdm

class DINOv2FeatureExtractor:
    def __init__(self, model_size='vits14'):
        """
        Khởi tạo mô hình DINOv2 từ torch.hub.
        model_size có thể là: 'vits14', 'vitb14', 'vitl14', 'vitg14'
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Loading DINOv2 model: dinov2_{model_size} on {self.device}...")
        
        # Load pre-trained model từ Meta
        self.model = torch.hub.load('facebookresearch/dinov2', f'dinov2_{model_size}')
        self.model = self.model.to(self.device)
        self.model.eval() # Set mode evaluation để tắt Dropout/BatchNorm

    def extract_features(self, dataloader):
        """
        Trích xuất đặc trưng cho toàn bộ dataset.
        """
        all_features = []
        all_labels = []

        print("Extracting features (This might take a while)...")
        with torch.no_grad():
            for images, labels in tqdm(dataloader, desc="Extracting"):
                images = images.to(self.device)
                
                # Forward pass qua DINOv2
                # DINOv2 trả về vector embedding cho mỗi ảnh (class token)
                features = self.model(images)
                
                # Chuyển về CPU để lưu lại tránh tràn VRAM
                all_features.append(features.cpu())
                all_labels.append(labels)

        return torch.cat(all_features), torch.cat(all_labels)
