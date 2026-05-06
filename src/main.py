import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import json
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from data_loader import get_intel_image_dataloaders

def extract_features(model, dataloader, device):
    """
    Trích xuất đặc trưng từ mô hình DINOv2 cho toàn bộ dataset.
    """
    model.eval()
    features = []
    labels = []
    with torch.no_grad():
        for images, targets in tqdm(dataloader, desc="Extracting features"):
            images = images.to(device)
            output = model(images)
            features.append(output.cpu().numpy())
            labels.append(targets.numpy())
    return np.concatenate(features), np.concatenate(labels)

def train_classifier(X_train, y_train, device, model_save_name="model.pth"):
    """
    Huấn luyện mô hình Linear Classifier (Linear Probing) bằng PyTorch.
    Quá trình này sẽ lưu lại hàm loss ở mỗi epoch.
    """
    print("Training Linear Classifier with PyTorch...")
    
    # Chuyển dữ liệu Numpy sang Tensor
    X_train_t = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_t = torch.tensor(y_train, dtype=torch.long).to(device)
    
    # Xác định số lớp và chiều đặc trưng (ViT-Small có chiều 384)
    num_classes = len(np.unique(y_train))
    feature_dim = X_train.shape[1]
    
    # Định nghĩa Linear layer
    classifier = nn.Linear(feature_dim, num_classes).to(device)
    
    # Khai báo hàm loss (CrossEntropy) và bộ tối ưu (Adam)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(classifier.parameters(), lr=1e-3)
    
    # Tạo DataLoader cho quá trình train
    dataset = torch.utils.data.TensorDataset(X_train_t, y_train_t)
    loader = torch.utils.data.DataLoader(dataset, batch_size=256, shuffle=True)
    
    epochs = 50
    loss_history = []
    
    classifier.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch_X, batch_y in loader:
            optimizer.zero_grad()
            outputs = classifier(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
        avg_loss = epoch_loss / len(loader)
        loss_history.append(avg_loss)
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")
            
    # Lưu hàm loss ra file JSON để sau này có thể vẽ biểu đồ
    with open('loss_history.json', 'w') as f:
        json.dump(loss_history, f)
    print("Đã lưu lịch sử hàm loss (Loss history) vào file 'loss_history.json'")
    
    # Lưu trọng số mô hình
    torch.save(classifier.state_dict(), model_save_name)
    print(f"Đã lưu trọng số mô hình phân loại vào file '{model_save_name}'")
    
    return classifier

def evaluate_classifier(classifier, X_test, y_test, device, dataset_name="Dataset"):
    """
    Đánh giá mô hình.
    """
    print("Evaluating...")
    classifier.eval()
    X_test_t = torch.tensor(X_test, dtype=torch.float32).to(device)
    
    with torch.no_grad():
        outputs = classifier(X_test_t)
        _, predicted = torch.max(outputs, 1)
        y_pred = predicted.cpu().numpy()
        
    acc = accuracy_score(y_test, y_pred)
    
    print(f"\n--- Kết quả (Results) ---")
    print(f"Độ chính xác (Accuracy) trên tập kiểm thử {dataset_name}: {acc * 100:.2f}%")
    return acc

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 1. Tải mô hình DINOv2 từ torch.hub
    print("Loading DINOv2 model (ViT-Small)...")
    model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
    model = model.to(device)

    # 2. Chuẩn bị dữ liệu
    print("Loading DataLoaders...")
    train_loader, test_loader, classes = get_intel_image_dataloaders(batch_size=64, num_workers=2)

    # 3. Trích xuất đặc trưng
    print("Extracting features for training set...")
    X_train, y_train = extract_features(model, train_loader, device)
    
    print("Extracting features for testing set...")
    X_test, y_test = extract_features(model, test_loader, device)

    # 4. Huấn luyện Linear Classifier (PyTorch)
    clf = train_classifier(X_train, y_train, device, model_save_name="model_intel_image.pth")

    # 5. Đánh giá kết quả
    evaluate_classifier(clf, X_test, y_test, device, dataset_name="Intel Image")

if __name__ == '__main__':
    main()
