import torch
import torch.nn as nn
import torchvision.models as models

class Dinov2LinearProbe(nn.Module):
    """
    Mô hình ứng dụng cơ chế Linear Probing trên đặc trưng của DINOv2.
    - Linear Probing: Đóng băng toàn bộ mạng lớn, chỉ huấn luyện một lớp tuyến tính cuối cùng.
    """
    def __init__(self, num_classes=7, freeze_backbone=True):
        super().__init__()
        # Tải bộ trọng số pre-trained DINOv2 (ViT-Base, patch size 14) từ PyTorch Hub của Meta
        self.backbone = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
        
        # Đóng băng (Freeze) các lớp của DINOv2 để không cập nhật trọng số trong lúc huấn luyện
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
                
        # DINOv2 ViT-B/14 tạo ra vector đặc trưng có chiều dài 768
        embed_dim = self.backbone.embed_dim
        # Khởi tạo lớp Linear duy nhất đóng vai trò làm bộ phân loại (Classifier)
        self.head = nn.Linear(embed_dim, num_classes)
        
    def forward(self, x):
        # Trích xuất đặc trưng hình ảnh bằng DINOv2
        features = self.backbone(x)
        # Đưa vector đặc trưng qua lớp Linear để phân loại
        return self.head(features)

class VitBaseline(nn.Module):
    """
    Mô hình tham chiếu (Baseline) sử dụng kiến trúc Vision Transformer (ViT-B/16).
    Mô hình này sử dụng trọng số pre-trained trên ImageNet (Supervised).
    """
    def __init__(self, num_classes=7, pretrained=True, freeze_backbone=True):
        super().__init__()
        # Tải mô hình ViT-B_16. Sử dụng trọng số có sẵn (ImageNet) nếu pretrained=True
        weights = models.ViT_B_16_Weights.DEFAULT if pretrained else None
        self.backbone = models.vit_b_16(weights=weights)
        
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
                
        # Lấy kích thước đầu vào của lớp Classification Head cuối cùng trong ViT
        num_ftrs = self.backbone.heads.head.in_features
        # Thay thế lớp này bằng một lớp mới phù hợp với số lượng class của SkinCancerMNIST (7)
        self.backbone.heads.head = nn.Linear(num_ftrs, num_classes)
        
    def forward(self, x):
        return self.backbone(x)

if __name__ == "__main__":
    # Đoạn code kiểm tra nhanh (Sanity checks) kiến trúc mạng và kích thước Tensor đầu ra
    dummy_input = torch.randn(2, 3, 224, 224)
    
    print("Testing DINOv2...")
    model_dino = Dinov2LinearProbe(num_classes=7)
    out_dino = model_dino(dummy_input)
    print("DINOv2 output shape:", out_dino.shape) # Output mong đợi: [2, 7]
    
    print("Testing ViT (ImageNet)...")
    model_vit = VitBaseline(num_classes=7)
    out_vit = model_vit(dummy_input)
    print("ViT output shape:", out_vit.shape) # Output mong đợi: [2, 7]
