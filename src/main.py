import os
from data_loader import get_dataloaders
from feature_extractor import DINOv2FeatureExtractor
from classifier import evaluate_knn
import torch

def main():
    print("="*50)
    print(" BẮT ĐẦU DỰ ÁN DINOV2 - VISUAL FEATURE EXTRACTION")
    print("="*50)
    
    # Cấu hình
    BATCH_SIZE = 64
    MODEL_SIZE = 'vits14' # Dùng ViT-Small để test nhanh, máy khỏe có thể đổi thành 'vitb14'
    K_NEIGHBORS = 20
    
    # 1. Load data
    print("\n[BƯỚC 1] Chuẩn bị dữ liệu (CIFAR-10)")
    train_loader, test_loader, classes = get_dataloaders(batch_size=BATCH_SIZE)
    print(f"Số lượng lớp (classes): {len(classes)} - {classes}")
    
    # 2. Khởi tạo mô hình DINOv2
    print("\n[BƯỚC 2] Tải mô hình DINOv2")
    extractor = DINOv2FeatureExtractor(model_size=MODEL_SIZE)
    
    # 3. Trích xuất đặc trưng
    print("\n[BƯỚC 3] Trích xuất đặc trưng từ ảnh")
    
    # Tùy chọn lưu feature để chạy lại nhanh hơn
    feature_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'features')
    os.makedirs(feature_dir, exist_ok=True)
    
    train_feat_path = os.path.join(feature_dir, f'train_feat_{MODEL_SIZE}.pt')
    test_feat_path = os.path.join(feature_dir, f'test_feat_{MODEL_SIZE}.pt')
    
    if os.path.exists(train_feat_path) and os.path.exists(test_feat_path):
        print("Đã tìm thấy file features lưu sẵn. Đang tải...")
        train_data = torch.load(train_feat_path)
        test_data = torch.load(test_feat_path)
        train_features, train_labels = train_data['features'], train_data['labels']
        test_features, test_labels = test_data['features'], test_data['labels']
    else:
        print("Trích xuất cho tập Train:")
        train_features, train_labels = extractor.extract_features(train_loader)
        
        print("Trích xuất cho tập Test:")
        test_features, test_labels = extractor.extract_features(test_loader)
        
        # Lưu lại
        torch.save({'features': train_features, 'labels': train_labels}, train_feat_path)
        torch.save({'features': test_features, 'labels': test_labels}, test_feat_path)
        print(f"Đã lưu features tại thư mục {feature_dir}")
        
    print(f"Kích thước feature vector: {train_features.shape[1]}")
    
    # 4. Đánh giá bằng K-NN
    print("\n[BƯỚC 4] Đánh giá chất lượng đặc trưng bằng K-NN")
    acc, report = evaluate_knn(train_features, train_labels, test_features, test_labels, k=K_NEIGHBORS)
    
    print("\n" + "="*50)
    print(" KẾT QUẢ ĐÁNH GIÁ (TEST SET)")
    print("="*50)
    print(f"Accuracy (K={K_NEIGHBORS}): {acc * 100:.2f}%")
    print("\nClassification Report:\n")
    print(report)

if __name__ == '__main__':
    main()
