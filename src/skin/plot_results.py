import os
import json
import matplotlib.pyplot as plt
import numpy as np

def load_history(model_name):
    """
    Hàm hỗ trợ đọc file JSON lịch sử kết quả của từng mô hình.
    """
    path = f"logs/{model_name}_history.json"
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return None

def plot_convergence():
    """
    Hàm vẽ biểu đồ đường (Line chart) để quan sát tốc độ hội tụ của mô hình (Dựa trên Validation Loss).
    Giúp chứng minh nhận định DINOv2 hội tụ rất nhanh nhờ đặc trưng có sẵn.
    """
    models = ["resnet50", "clip", "dinov2"]
    colors = {"resnet50": "blue", "clip": "green", "dinov2": "red"}
    labels = {"resnet50": "ResNet50 (Supervised)", "clip": "CLIP (Zero-shot/Linear)", "dinov2": "DINOv2 (Linear Probe)"}
    
    plt.figure(figsize=(10, 6))
    
    # Đọc kết quả và vẽ lên 1 biểu đồ duy nhất
    for m in models:
        hist = load_history(m)
        if hist:
            val_loss = hist["val_loss"]
            plt.plot(range(1, len(val_loss)+1), val_loss, marker='o', color=colors[m], label=labels[m])
            
    plt.title("Đồ thị hội tụ (Validation Loss)", fontsize=14)
    plt.xlabel("Epochs", fontsize=12)
    plt.ylabel("Validation Loss", fontsize=12)
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.7)
    
    # Lưu ra thư mục report để LaTeX tự động cập nhật
    plt.savefig("report/figures/convergence.png", dpi=300, bbox_inches='tight')
    print("Saved convergence.png")

def plot_bar_metrics():
    """
    Hàm vẽ biểu đồ cột (Bar chart) so sánh trực tiếp hiệu năng cuối cùng trên tập Test
    dựa trên 2 tiêu chí cốt lõi: Accuracy và F1-Score.
    """
    models = ["resnet50", "clip", "dinov2"]
    labels = ["ResNet50\n(Supervised)", "CLIP\n(Linear)", "DINOv2\n(Linear)"]
    
    accs = []
    f1s = []
    valid_labels = []
    
    # Lọc ra những mô hình đã có kết quả
    for m, l in zip(models, labels):
        hist = load_history(m)
        if hist:
            accs.append(hist["test_acc"])
            f1s.append(hist["test_f1"])
            valid_labels.append(l)
            
    if not accs:
        print("Chưa có dữ liệu nào để vẽ biểu đồ cột.")
        return
        
    x = np.arange(len(valid_labels))
    width = 0.35 # Độ rộng của cột
    
    fig, ax = plt.subplots(figsize=(8, 6))
    # Vẽ cột Accuracy
    rects1 = ax.bar(x - width/2, accs, width, label='Accuracy', color='skyblue')
    # Vẽ cột F1-Score
    rects2 = ax.bar(x + width/2, f1s, width, label='F1-Score', color='lightcoral')
    
    ax.set_ylabel('Scores', fontsize=12)
    ax.set_title('So sánh hiệu năng các mô hình trên SkinCancerMNIST', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(valid_labels, fontsize=11)
    ax.legend(loc='lower right')
    ax.set_ylim([0, 1.1]) # Chặn trục Y từ 0 đến 1.1 để đồ thị thoáng
    
    # Hàm con tự động hiển thị số (giá trị) trên đỉnh từng cột
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.3f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # Đẩy lên 3 points
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=10)
    
    autolabel(rects1)
    autolabel(rects2)
    
    fig.tight_layout()
    # Lưu ra thư mục report để LaTeX tự động cập nhật
    plt.savefig("report/figures/performance_bar.png", dpi=300)
    print("Saved performance_bar.png")

def plot_confusion_matrix():
    """
    Hàm vẽ Confusion Matrix cho từng mô hình đã chạy.
    """
    import sys
    import os
    # Đảm bảo có thể import data_loader
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from data_loader import DX_CLASSES
    from sklearn.metrics import ConfusionMatrixDisplay
    
    models = ["resnet50", "clip", "dinov2"]
    
    for m in models:
        hist = load_history(m)
        if hist and "test_cm" in hist:
            cm = np.array(hist["test_cm"])
            fig, ax = plt.subplots(figsize=(8, 6))
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=DX_CLASSES)
            disp.plot(cmap="Blues", ax=ax, values_format="d")
            plt.title(f"Confusion Matrix - {m.upper()}", fontsize=14)
            plt.tight_layout()
            
            save_path = f"report/figures/confusion_matrix_{m}.png"
            plt.savefig(save_path, dpi=300)
            print(f"Saved {save_path}")
            plt.close()

if __name__ == "__main__":
    os.makedirs("report/figures", exist_ok=True)
    plot_convergence()
    plot_bar_metrics()
    plot_confusion_matrix()
