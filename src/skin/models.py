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

class ResNet50Baseline(nn.Module):
    """
    Mô hình tham chiếu (Baseline) sử dụng kiến trúc ResNet50.
    Mô hình này sẽ được Fine-tune toàn bộ có giám sát.
    """
    def __init__(self, num_classes=7, pretrained=True):
        super().__init__()
        # Tải mô hình ResNet50. Sử dụng trọng số có sẵn (ImageNet) nếu pretrained=True
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        self.backbone = models.resnet50(weights=weights)
        
        # Lấy kích thước đầu vào của lớp Fully Connected cuối cùng trong ResNet50
        num_ftrs = self.backbone.fc.in_features
        # Thay thế lớp này bằng một lớp mới phù hợp với số lượng class của SkinCancerMNIST (7)
        self.backbone.fc = nn.Linear(num_ftrs, num_classes)
        
    def forward(self, x):
        return self.backbone(x)

class ClipLinearProbe(nn.Module):
    """
    Mô hình ứng dụng Linear Probing trên Image Encoder của CLIP (OpenCLIP).
    """
    def __init__(self, num_classes=7, freeze_backbone=True):
        super().__init__()
        import open_clip
        # Tải Image Encoder của CLIP với kiến trúc ViT-B/32, được huấn luyện trên tập LAION-2B
        model, _, _ = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
        self.backbone = model.visual # Chỉ lấy phần Visual (Image Encoder), bỏ phần Text Encoder
        
        # Tương tự như DINOv2, đóng băng mạng Backbone
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
                
        # Lấy kích thước đầu ra của CLIP (thường là 512)
        embed_dim = self.backbone.output_dim
        self.head = nn.Linear(embed_dim, num_classes)
        
    def forward(self, x):
        features = self.backbone(x)
        return self.head(features)

if __name__ == "__main__":
    # Đoạn code kiểm tra nhanh (Sanity checks) kiến trúc mạng và kích thước Tensor đầu ra
    dummy_input = torch.randn(2, 3, 224, 224)
    
    print("Testing DINOv2...")
    model_dino = Dinov2LinearProbe(num_classes=7)
    out_dino = model_dino(dummy_input)
    print("DINOv2 output shape:", out_dino.shape) # Output mong đợi: [2, 7]
    
    print("Testing ResNet50...")
    model_resnet = ResNet50Baseline(num_classes=7)
    out_resnet = model_resnet(dummy_input)
    print("ResNet50 output shape:", out_resnet.shape) # Output mong đợi: [2, 7]
    
    print("Testing CLIP...")
    model_clip = ClipLinearProbe(num_classes=7)
    out_clip = model_clip(dummy_input)
    print("CLIP output shape:", out_clip.shape) # Output mong đợi: [2, 7]
