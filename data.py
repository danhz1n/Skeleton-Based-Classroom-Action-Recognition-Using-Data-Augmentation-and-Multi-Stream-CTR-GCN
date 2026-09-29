import pickle
import numpy as np
import os

# 1. Đường dẫn file tập test hiện tại của bạn
test_data_path = r"C:\Users\User\CTR-GCN\data\dav\EduAction_pose_data\test_data.pkl"
output_data_path = r"C:\Users\User\CTR-GCN\data\dav\EduAction_pose_data\test_data_norm.pkl"

print("Đang đọc dữ liệu tập test...")
with open(test_data_path, 'rb') as f:
    X_test = pickle.load(f)

# Kích thước dữ liệu mặc định là (N, C, T, V, M) với C=2 (X, Y)
print(f"Kích thước ban đầu: {X_test.shape}")
print(f"Giá trị X max ban đầu: {X_test[:, 0].max():.2f}")
print(f"Giá trị Y max ban đầu: {X_test[:, 1].max():.2f}")

# 2. Xử lý chuẩn hóa
# Tính giá trị tọa độ lớn nhất (để chia lấy tỷ lệ giữ nguyên Aspect Ratio)
max_val = np.max(X_test)

if max_val < 10:
    print("Dữ liệu có vẻ đã được chuẩn hóa từ trước, không cần làm gì thêm!")
else:
    # Chia toàn bộ tọa độ cho max_val để đưa về khoảng [0, 1]
    X_test_norm = X_test / max_val
    
    print(f"\nĐã chuẩn hóa xong!")
    print(f"Giá trị X max mới: {X_test_norm[:, 0].max():.2f}")
    print(f"Giá trị Y max mới: {X_test_norm[:, 1].max():.2f}")
    
    # 3. Lưu thành file mới
    with open(output_data_path, 'wb') as f:
        pickle.dump(X_test_norm, f)
    
    print(f"\n🎉 Đã lưu tập test mới (đã chuẩn hóa) tại:\n-> {output_data_path}")
    print("⚠️ BƯỚC TIẾP THEO: Hãy vào file config.yaml và sửa dòng data_path của test_feeder_args thành 'test_data_norm.pkl' nhé!")