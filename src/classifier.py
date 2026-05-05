from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
import time

def evaluate_knn(train_features, train_labels, test_features, test_labels, k=20):
    """
    Đánh giá đặc trưng sử dụng K-Nearest Neighbors classifier.
    Đây là cách phổ biến để đánh giá chất lượng đặc trưng (features)
    của các mô hình Self-Supervised Learning mà không cần fine-tune.
    """
    print(f"\nTraining K-NN classifier with K={k}...")
    start_time = time.time()
    
    knn = KNeighborsClassifier(n_neighbors=k, n_jobs=-1) # Dùng n_jobs=-1 để tận dụng đa luồng
    
    # Huấn luyện K-NN bằng cách fit các features
    knn.fit(train_features.numpy(), train_labels.numpy())
    
    print(f"Training completed in {time.time() - start_time:.2f} seconds.")
    
    # Dự đoán trên tập Test
    print("Evaluating on test set...")
    start_time = time.time()
    predictions = knn.predict(test_features.numpy())
    print(f"Evaluation completed in {time.time() - start_time:.2f} seconds.")
    
    # Tính toán các metrics
    acc = accuracy_score(test_labels.numpy(), predictions)
    report = classification_report(test_labels.numpy(), predictions)
    
    return acc, report
