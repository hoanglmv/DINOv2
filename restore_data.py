import json
import re

# 1. Restore JSON
train_loss = [0.7619, 0.6103, 0.5746, 0.5370, 0.5103, 0.4934, 0.4810, 0.4847, 0.4752, 0.4608, 0.4565, 0.4481, 0.4369, 0.4313, 0.4223, 0.4371]
train_acc = [0.7280, 0.7729, 0.7882, 0.7990, 0.8101, 0.8141, 0.8262, 0.8234, 0.8270, 0.8304, 0.8334, 0.8354, 0.8384, 0.8397, 0.8399, 0.8352]
val_loss = [0.6782, 0.5958, 0.5625, 0.5698, 0.5501, 0.5351, 0.5277, 0.5274, 0.5272, 0.5830, 0.5117, 0.5556, 0.5072, 0.5426, 0.5280, 0.6394]
val_acc = [0.7315, 0.7774, 0.7784, 0.7934, 0.7934, 0.7984, 0.8054, 0.8064, 0.7994, 0.7974, 0.8224, 0.7874, 0.8124, 0.8154, 0.8034, 0.7595]

try:
    with open("logs/dinov2_history.json", "r") as f:
        old_data = json.load(f)
except Exception:
    old_data = {"test_cm": []}

new_data = {
    "train_loss": train_loss,
    "train_acc": train_acc,
    "val_loss": val_loss,
    "val_acc": val_acc,
    "test_acc": 0.8128,
    "test_f1": 0.8058,
    "test_cm": old_data.get("test_cm", [])
}

with open("logs/dinov2_history.json", "w") as f:
    json.dump(new_data, f)

# 2. Update chapter2.tex
with open("report/chapter2/chapter2.tex", "r") as f:
    content = f.read()

content = content.replace(r"\textbf{78.08} & \textbf{77.24}", r"\textbf{81.28} & \textbf{80.58}")
content = content.replace("DINOv2 đạt 78.08\\% Accuracy và 77.24\\% F1-Score.", "DINOv2 đạt 81.28\\% Accuracy và 80.58\\% F1-Score.")
content = content.replace("Dù thấp hơn Baseline khoảng 4.55\\%,", "Dù thấp hơn Baseline khoảng 1.35\\%,")

# Fix early stopping paragraph
old_para = r"Như quan sát trên Hình \ref{fig:convergence}, mô hình DINOv2 (đường màu đỏ) cho thấy tốc độ hội tụ cực kỳ ổn định và nhanh chóng trong những epoch đầu tiên. Đáng chú ý, đồ thị hội tụ của DINOv2 dừng lại ở epoch thứ 7 thay vì chạy hết 20 epochs. Điều này là kết quả trực tiếp của cơ chế Early Stopping (với patience=5) đã được thiết lập. Do DINOv2 trích xuất đặc trưng quá tốt, Validation Loss đã chạm mức cực tiểu rất sớm và không cải thiện thêm, nên hệ thống đã chủ động dừng quá trình huấn luyện nhằm ngăn chặn hiện tượng quá khớp (Overfitting) cũng như tối ưu thời gian. Mô hình ViT tham chiếu (đường màu xanh) hội tụ chậm hơn nên tiếp tục chạy lâu hơn. Điều này chứng tỏ bộ đặc trưng ban đầu mà DINOv2 cung cấp nhờ phương pháp tự giám sát đã chứa sẵn hàm lượng thông tin tổng quát cực kỳ cao, cho phép lớp Linear Head phân tách không gian đa chiều một cách dễ dàng so với các đặc trưng ImageNet của ViT."
new_para = r"Như quan sát trên Hình \ref{fig:convergence}, mô hình DINOv2 (đường màu đỏ) cho thấy tốc độ hội tụ cực kỳ ổn định và nhanh chóng trong những epoch đầu tiên. Đáng chú ý, đồ thị hội tụ của DINOv2 dừng lại ở epoch thứ 16 thay vì chạy hết 20 epochs. Điều này là kết quả trực tiếp của cơ chế Early Stopping (với patience=5) đã được thiết lập. Do DINOv2 đã đạt mức Loss cực tiểu trên tập Validation ở epoch 11 và không cải thiện thêm trong 5 epoch sau đó, hệ thống đã chủ động dừng quá trình huấn luyện nhằm ngăn chặn hiện tượng quá khớp (Overfitting). Mô hình ViT tham chiếu (đường màu xanh) hội tụ chậm hơn nên tiếp tục chạy trọn vẹn 20 epochs. Điều này chứng tỏ bộ đặc trưng ban đầu mà DINOv2 cung cấp nhờ phương pháp tự giám sát đã chứa sẵn hàm lượng thông tin tổng quát cực kỳ cao, cho phép lớp Linear Head phân tách không gian đa chiều một cách dễ dàng so với các đặc trưng ImageNet của ViT."
content = content.replace(old_para, new_para)

with open("report/chapter2/chapter2.tex", "w") as f:
    f.write(content)

print("Data restored and chapter2 updated.")
