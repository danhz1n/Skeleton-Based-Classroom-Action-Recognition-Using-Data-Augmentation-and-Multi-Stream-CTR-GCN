import os
import pickle
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix

# Khởi tạo trực tiếp kiến trúc mô hình và bộ nạp dữ liệu từ source code gốc
from model.ctrgcn import Model
from feeders.feeder_dav import Feeder

# ==============================================================================
# 1. CẤU HÌNH ĐƯỜNG DẪN VÀ THAM SỐ THỰC TẾ (6 CLASSES | 17 POINTS)
# ==============================================================================
CHECKPOINT_PATH = r"C:\Users\User\CTR-GCN\work_dir\dav\ctrgcn_joint\runs-25-9300.pt" 
TEST_DATA_PATH = r"./data/dav/test_data.pkl" 
TEST_LABEL_PATH = r"./data/dav/test_label.pkl" 
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Tên 6 hành động thực tế khớp 100% với ID nhãn từ 0 đến 5
class_names = ["drinking", "play_phone", "sleeping", "talking", "watch_computer", "writing"]
num_classes = len(class_names)

# ==============================================================================
# 2. KHỞI TẠO MODEL TRỰC TIẾP & ĐỔ TRỌNG SỐ (CHECKPOINT)
# ==============================================================================
print("-> Đang khởi tạo kiến trúc mạng CTR-GCN thuần túy...")

# Dựng trực tiếp mô hình mà không cần thông qua class Processor của main.py
model = Model(
    num_class=num_classes,   # 6 lớp hành động
    num_point=17,            # 17 khớp xương
    num_person=2,            # Hỗ trợ tối đa 2 người
    graph='graph.dav.Graph',
    graph_args={'labeling_mode': 'spatial'}
)

if os.path.exists(CHECKPOINT_PATH):
    # Nạp ma trận trọng số .pt đã huấn luyện vào mô hình
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint)
    print(f"✅ Đã nạp thành công trọng số từ: {CHECKPOINT_PATH}")
else:
    print(f"🚨 [LỖI]: Không tìm thấy file trọng số tại {CHECKPOINT_PATH}!")
    exit()

model = model.to(DEVICE)
model.eval()

# ==============================================================================
# 3. NẠP TẬP DỮ LIỆU KIỂM THỬ (TEST LOAD)
# ==============================================================================
print("-> Đang nạp tập dữ liệu Test độc lập...")
dataset = Feeder(data_path=TEST_DATA_PATH, label_path=TEST_LABEL_PATH)
test_loader = DataLoader(dataset, batch_size=16, shuffle=False)

# ==============================================================================
# 4. TIẾN TRÌNH DỰ ĐOÁN (INFERENCE LUỒNG TEST)
# ==============================================================================
all_preds = []
all_labels = []

print("-> Đang chạy dự đoán trên toàn bộ tập Test...")
with torch.no_grad():
    for batch in test_loader:
        data, label = batch[0], batch[1]
        data = data.float().to(DEVICE)
        
        output = model(data)
        pred = output.argmax(dim=1)
        
        all_preds.extend(pred.cpu().numpy())
        all_labels.extend(label.numpy())

all_preds = np.array(all_preds)
all_labels = np.array(all_labels)

# ==============================================================================
# 5. TÍNH TOÁN VÀ CHUẨN HÓA MA TRẬN NHẦM LẪN (CONFUSION MATRIX)
# ==============================================================================
cm = confusion_matrix(all_labels, all_preds, labels=range(num_classes))

# Chuẩn hóa theo hàng (Tỷ lệ phần trăm phân loại đúng/sai của từng class thực tế)
cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-6)

# ==============================================================================
# 6. VẼ VÀ TRỰC QUAN HÓA ĐỒ THỊ HEATMAP
# ==============================================================================
print("-> Đang dựng và vẽ biểu đồ Heatmap...")
fig, axes = plt.subplots(1, 2, figsize=(20, 8), dpi=150)
sns.set_theme(style="white")

# Biểu đồ 1: Số lượng mẫu thực tế (Raw counts)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True,
            xticklabels=class_names, yticklabels=class_names, ax=axes[0])
axes[0].set_title('Confusion Matrix (Raw Sample Counts)', fontsize=14, fontweight='bold', pad=10)
axes[0].set_xlabel('Predicted Action (Mô hình đoán)', fontsize=12)
axes[0].set_ylabel('True Action (Thực tế)', fontsize=12)
axes[0].tick_params(axis='x', rotation=30)

# Biểu đồ 2: Tỷ lệ phần trăm chuẩn hóa (Normalized)
sns.heatmap(cm_norm, annot=True, fmt='.1%', cmap='OrRd', cbar=True, vmin=0, vmax=1,
            xticklabels=class_names, yticklabels=class_names, ax=axes[1])
axes[1].set_title('Confusion Matrix (Normalized Accuracy Rate)', fontsize=14, fontweight='bold', pad=10)
axes[1].set_xlabel('Predicted Action (Mô hình đoán)', fontsize=12)
axes[1].set_ylabel('True Action (Thực tế)', fontsize=12)
axes[1].tick_params(axis='x', rotation=30)

plt.tight_layout()
output_fig_path = './work_dir/dav/ctrgcn_joint/confusion_matrix_evaluation.png'
os.makedirs(os.path.dirname(output_fig_path), exist_ok=True)
plt.savefig(output_fig_path, bbox_inches='tight')
plt.show()
print(f"✅ Biểu đồ ma trận đã được lưu an toàn tại: {output_fig_path}")

# ==============================================================================
# 7. IN THỐNG KÊ CÁC CẶP NHẦM LẪN NGHIÊM TRỌNG NHẤT KHẢO SÁT
# ==============================================================================
print("\n" + "="*20 + " TOP CÁC CẶP HÀNH ĐỘNG BỊ NHẦM LẪN NHIỀU NHẤT " + "="*20)
print(f"{'Hành động Gốc (True)':>22}  -->  {'Bị Nhầm Sang (Pred)':>22}  |  {'Số lần':>8}  |  {'Tỷ lệ lỗi':>10}")
print("-" * 75)

errors = []
for i in range(num_classes):
    for j in range(num_classes):
        if i != j and cm[i, j] > 0:
            errors.append((i, j, cm[i, j], cm_norm[i, j]))

errors.sort(key=lambda x: x[2], reverse=True)

if len(errors) == 0:
    print("🎉 Tuyệt vời! Không có hành động nào bị phân loại nhầm trên tập kiểm thử.")
else:
    for true, pred, count, rate in errors[:10]:
        print(f"{class_names[true]:>22}  -->  {class_names[pred]:>22}  |  {count:>6}x   |  {rate:>9.1%}")

# ==============================================================================
# 8. THỐNG KÊ ĐỘ CHÍNH XÁC CHI TIẾT TỪNG LỚP (PER-CLASS ACCURACY)
# ==============================================================================
print("\n" + "="*25 + " ĐỘ CHÍNH XÁC THEO TỪNG HÀNH ĐỘNG " + "="*25)
for i in range(num_classes):
    acc = cm_norm[i, i]
    bar_length = int(acc * 20)
    bar = '█' * bar_length + '░' * (20 - bar_length)
    print(f"  - {class_names[i]:>15}:  [{bar}]  Độ chính xác: {acc:.1%}")
print("=" * 80)