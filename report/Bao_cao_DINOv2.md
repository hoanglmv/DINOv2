# Báo cáo Dự án: Tìm hiểu mô hình DINOv2
**Bài báo:** DINOv2: Learning Robust Visual Features without Supervision

## 1. Tóm tắt nội dung chính của Paper

### 1.1. Problem (Bài toán)
Trong lĩnh vực thị giác máy tính (Computer Vision), việc học các đặc trưng hình ảnh (visual features) mạnh mẽ và có thể tổng quát hóa cho nhiều tác vụ (image classification, segmentation, depth estimation) thường phụ thuộc rất lớn vào lượng lớn dữ liệu được gán nhãn thủ công. Quá trình gán nhãn này rất tốn kém và không thể mở rộng (scale-up). Bài toán đặt ra là làm thế nào để huấn luyện một mô hình có khả năng trích xuất đặc trưng hình ảnh chất lượng cao, đa dụng (all-purpose) mà **không cần sử dụng nhãn dữ liệu** (unsupervised/self-supervised learning), tương tự như cách các mô hình ngôn ngữ lớn (LLMs) đang làm với văn bản.

### 1.2. Motivation (Động lực)
- Các mô hình Transformer trong xử lý ngôn ngữ tự nhiên (NLP) đã chứng minh rằng: huấn luyện trên lượng dữ liệu không gán nhãn khổng lồ có thể tạo ra các đặc trưng (features) đủ tốt để sử dụng cho mọi tác vụ downstream (zero-shot, few-shot).
- Trong Computer Vision, dù các phương pháp tự giám sát (Self-Supervised Learning - SSL) đã có nhiều tiến bộ (như DINOv1, iBOT), nhưng chúng chủ yếu được huấn luyện trên các dataset có cấu trúc như ImageNet-1K, vốn thiếu tính đa dạng. 
- Động lực chính của nhóm tác giả (Meta AI) là xây dựng một hệ thống pipeline tự động để thu thập, tinh lọc (curate) một tập dữ liệu khổng lồ, đa dạng, không cần nhãn (dataset LVD-142M) và tối ưu hóa kiến trúc ViT (Vision Transformer) để scale mô hình lên kích thước cực lớn, nhằm tạo ra một mô hình nền tảng (Foundation Model) cho thị giác máy tính thực thụ.

### 1.3. Method (Phương pháp)
DINOv2 là sự kết hợp của nhiều cải tiến về cả mặt **dữ liệu** và **thuật toán huấn luyện**:

- **Về Dữ liệu (LVD-142M Dataset):**
  - Xây dựng một pipeline tự động để lọc ra các hình ảnh chất lượng cao từ các nguồn web chưa được giám định.
  - Sử dụng kỹ thuật thu hồi hình ảnh (Image Retrieval) để lấy ra những hình ảnh từ nguồn web chưa chọn lọc sao cho phân phối của chúng tương đồng với các tập dữ liệu chất lượng cao đã biết (như ImageNet-22k, in-domain data). Kết quả tạo ra bộ dữ liệu LVD-142M (142 triệu ảnh).
- **Về Thuật toán (Discriminative Self-supervised Learning):**
  - Kết hợp hai phương pháp mạnh nhất hiện tại là **DINO** (Self-distillation với loss đối xứng trên các bản crop khác nhau của cùng một ảnh) và **iBOT** (Masked Image Modeling - tương tự BERT, che đi một số patch của ảnh và yêu cầu mô hình dự đoán lại nội dung đó).
  - Áp dụng các kỹ thuật tối ưu hóa mô hình ở quy mô lớn: Sử dụng FlashAttention, Fully Sharded Data Parallel (FSDP), Stochastic Depth, và kỹ thuật chưng cất mô hình (Model Distillation) để nén từ mô hình khổng lồ (ViT-g) xuống các mô hình nhỏ hơn (ViT-S, ViT-B, ViT-L).
- **Patch-level Objective và High-resolution:** Thêm kỹ thuật Sinkhorn-Knopp để cân bằng center và tăng cường độ phân giải hình ảnh trong giai đoạn huấn luyện cuối để cải thiện khả năng trích xuất đặc trưng cho các task yêu cầu độ phân giải cao như segmentation hay depth estimation.

### 1.4. Result (Kết quả)
- **Image Classification:** DINOv2 đạt độ chính xác cực cao trên ImageNet-1K chỉ bằng việc phân loại tuyến tính (Linear evaluation) hoặc k-NN (K-Nearest Neighbors) mà không cần fine-tune toàn bộ mô hình (VD: mô hình lớn nhất ViT-g đạt 86.5% accuracy với linear probe).
- **Dense Tasks:** Vượt trội hơn đáng kể so với các mô hình SSL trước đây trên các tác vụ yêu cầu mức độ pixel (dense tasks) như Semantic Segmentation (Pascal VOC, ADE20K) và Monocular Depth Estimation (NYUv2), chứng tỏ các feature bảo toàn được rất tốt cấu trúc không gian (spatial layout).
- **Zero-shot / Freeze Features:** Mô hình chứng minh được khả năng ứng dụng trực tiếp "Out-of-the-box". Các feature của nó có thể được dùng làm input trực tiếp cho các bộ phân loại tuyến tính đơn giản mà vẫn đạt kết quả SOTA.

### 1.5. Contribution (Đóng góp)
1. **Mô hình nền tảng CV:** Cung cấp bộ các mô hình DINOv2 (ViT-S, ViT-B, ViT-L, ViT-g) có khả năng sinh ra feature hình ảnh vạn năng, tốt nhất hiện tại cho kiến trúc học tự giám sát.
2. **Dữ liệu:** Đề xuất một pipeline hoàn chỉnh để tinh lọc dữ liệu ảnh chưa gán nhãn quy mô khổng lồ thành dữ liệu chất lượng cao (LVD-142M).
3. **Kỹ thuật tối ưu:** Tập hợp các kỹ thuật thiết kế và huấn luyện hiệu quả giúp ổn định quá trình scale-up mạng ViT lên quy mô cực lớn mà không bị sụp đổ (collapse) trong quá trình học SSL.
4. **Mở mã nguồn:** Toàn bộ weights và code được cung cấp miễn phí để cộng đồng sử dụng.

---
## 2. Thực nghiệm Code (Đang thực hiện)
- Dataset: [Đang chọn]
- Bài toán: Phân loại ảnh với K-NN

---
## 3. Đánh giá và Critical Thinking (Chưa hoàn thiện)
- Sẽ bổ sung sau khi có kết quả thực nghiệm.
