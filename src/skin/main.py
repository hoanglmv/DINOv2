import os
import argparse
import json
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

from data_loader import get_dataloaders
from models import Dinov2LinearProbe, ResNet50Baseline, ClipLinearProbe

def train_epoch(model, dataloader, criterion, optimizer, device):
    """
    Thực hiện 1 Epoch (vòng lặp) huấn luyện trên tập Train.
    """
    model.train() # Chuyển mô hình sang chế độ huấn luyện (kích hoạt Dropout, BatchNorm)
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    # Dùng tqdm để hiển thị thanh tiến trình
    pbar = tqdm(dataloader, desc="Training")
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad() # Xóa bộ nhớ gradient cũ
        outputs = model(images) # Lan truyền tiến (Forward pass)
        loss = criterion(outputs, labels) # Tính toán Loss
        loss.backward() # Lan truyền ngược (Backward pass) để tính gradient
        optimizer.step() # Cập nhật trọng số
        
        running_loss += loss.item() * images.size(0)
        
        # Dự đoán nhãn (class có xác suất cao nhất)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
        # Cập nhật thanh tiến trình hiển thị Loss hiện tại
        pbar.set_postfix({"loss": loss.item()})
        
    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = accuracy_score(all_labels, all_preds)
    return epoch_loss, epoch_acc

def evaluate(model, dataloader, criterion, device):
    """
    Đánh giá mô hình trên tập Validation hoặc Test.
    """
    model.eval() # Chuyển mô hình sang chế độ đánh giá
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    # torch.no_grad() giúp tiết kiệm bộ nhớ, vì lúc này không cần tính gradient
    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Evaluating")
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * images.size(0)
            
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = accuracy_score(all_labels, all_preds)
    # Dùng macro/weighted f1-score vì tập dữ liệu y tế này mất cân bằng (imbalanced)
    epoch_f1 = f1_score(all_labels, all_preds, average='weighted')
    cm = confusion_matrix(all_labels, all_preds)
    return epoch_loss, epoch_acc, epoch_f1, cm

def main(args):
    # Cấu hình thiết bị (GPU nếu có, ngược lại dùng CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Sử dụng thiết bị: {device}")
    
    # 1. Chuẩn bị Dữ liệu (Data Pipeline)
    base_dir = "data/SkinCancer"
    csv_path = os.path.join(base_dir, "HAM10000_metadata.csv")
    img_dirs = [os.path.join(base_dir, "HAM10000_images_part_1"), os.path.join(base_dir, "HAM10000_images_part_2")]
    
    print(f"Đang tải dữ liệu cho mô hình: {args.model}")
    train_loader, val_loader, test_loader, num_classes = get_dataloaders(
        csv_path, img_dirs, model_type=args.model, batch_size=args.batch_size, num_workers=4
    )
    
    # 2. Khởi tạo Mô hình
    if args.model == "dinov2":
        model = Dinov2LinearProbe(num_classes=num_classes).to(device)
    elif args.model == "resnet50":
        model = ResNet50Baseline(num_classes=num_classes).to(device)
    elif args.model == "clip":
        model = ClipLinearProbe(num_classes=num_classes).to(device)
    else:
        raise ValueError(f"Không nhận diện được mô hình: {args.model}")
        
    # 3. Thiết lập thông số Huấn luyện
    criterion = nn.CrossEntropyLoss()
    # model.parameters() với DINOv2 và CLIP thì chỉ có trọng số của head là có requires_grad=True
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # Biến theo dõi kết quả để sau này vẽ đồ thị
    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": []
    }
    
    best_val_acc = 0.0
    os.makedirs("logs", exist_ok=True)
    os.makedirs("weights", exist_ok=True)
    
    # 4. Vòng lặp Huấn luyện chính (Training Loop)
    print(f"Bắt đầu huấn luyện với {args.epochs} epochs...")
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, device)
        
        # Lưu kết quả
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.4f}")
        
        # Chỉ lưu mô hình khi độ chính xác trên tập Validation tăng lên (Early stopping type)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), f"weights/{args.model}_best.pth")
            
    # 5. Đánh giá cuối cùng trên tập Test
    print("\nTải lại trọng số tốt nhất để chạy đánh giá trên tập Test...")
    model.load_state_dict(torch.load(f"weights/{args.model}_best.pth"))
    test_loss, test_acc, test_f1, test_cm = evaluate(model, test_loader, criterion, device)
    
    print(f"Accuracy trên tập Test: {test_acc:.4f}")
    print(f"F1-Score trên tập Test: {test_f1:.4f}")
    
    history["test_acc"] = test_acc
    history["test_f1"] = test_f1
    history["test_cm"] = test_cm.tolist()
    
    # Xuất lịch sử ra file JSON
    with open(f"logs/{args.model}_history.json", "w") as f:
        json.dump(history, f)
        
if __name__ == "__main__":
    # Cấu hình để truyền tham số bằng command line
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=["dinov2", "resnet50", "clip"], required=True, help="Tên mô hình")
    parser.add_argument("--batch_size", type=int, default=32, help="Kích thước batch")
    parser.add_argument("--epochs", type=int, default=10, help="Số epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    args = parser.parse_args()
    main(args)
