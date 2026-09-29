import os
import pickle
import numpy as np
import torch
import shap
import matplotlib.pyplot as plt

# Nhập kiến trúc mô hình và bộ nạp dữ liệu từ dự án CTR-GCN của bạn
from model.ctrgcn import Model
from feeders.feeder_dav import Feeder

# ==============================================================================
# 1. CẤU HÌNH ĐƯỜNG DẪN CHUẨN ĐỒNG BỘ THEO CONFIG TRAIN (6 CHANNELS)
# ==============================================================================
# ĐÃ SỬA: Trỏ đúng vào thư mục ctrgcn_dav theo biến work_dir trong log train của bạn
CHECKPOINT_PATH = r"C:\Users\User\CTR-GCN\work_dir\dav\ctrgcn_dav\runs-8-1488.pt" 
TEST_DATA_PATH = r"./data/dav/test_data.pkl" 
TEST_LABEL_PATH = r"./data/dav/test_label.pkl" 
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Tên 6 hành động thực tế khớp 100% với ID nhãn từ 0 đến 5
class_names = ["drinking", "play_phone", "sleeping", "talking", "watch_computer", "writing"]
num_classes = len(class_names)

# ==============================================================================
# 2. KHỞI TẠO MODEL TRỰC TIẾP & ĐỔ TRỌNG SỐ (CHECKPOINT)
# ==============================================================================
print("-> Đang nạp mô hình CTR-GCN...")

# ĐỒNG BỘ 100%: Cấu hình đúng 6 channels, 17 points, 1 person theo ảnh chụp Terminal
model = Model(
    num_class=num_classes,   
    num_point=17,            
    num_person=1,            
    in_channels=6,           
    graph='graph.dav.Graph',
    graph_args={'labeling_mode': 'spatial'}
).to(DEVICE)

if os.path.exists(CHECKPOINT_PATH):
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint)
    print("✅ Nạp trọng số thành công!")
else:
    print(f"🚨 [LỖI]: Không tìm thấy file trọng số tại {CHECKPOINT_PATH}!")
    print("Vui lòng kiểm tra lại số Epoch (ví dụ: runs-25-9300.pt hoặc runs-30-xxxx.pt) trong thư mục work_dir/dav/ctrgcn_dav/")
    exit()

model.eval()

# ==============================================================================
# 3. NẠP TẬP DỮ LIỆU KIỂM THỬ (TEST LOAD)
# ==============================================================================
print("-> Đang nạp mẫu dữ liệu...")
dataset = Feeder(data_path=TEST_DATA_PATH, label_path=TEST_LABEL_PATH)

# Trích xuất an toàn để tránh UserWarning và lỗi ép kiểu Tensor
bg_samples = []
for i in range(10):
    sample_data = dataset[i][0]
    if isinstance(sample_data, torch.Tensor):
        bg_samples.append(sample_data.clone().detach())
    else:
        bg_samples.append(torch.tensor(sample_data))
background_data = torch.stack(bg_samples).float().to(DEVICE)

# Hứng trọn vẹn 3 phần tử dữ liệu từ bộ nạp Feeder độc lập
sample_tensor, sample_label, _ = dataset[0]

# Giữ nguyên toàn bộ cấu trúc 6 kênh đặc trưng của mẫu dữ liệu đầu vào
input_tensor = torch.from_numpy(sample_tensor).float().unsqueeze(0).to(DEVICE) if isinstance(sample_tensor, np.ndarray) else sample_tensor.clone().detach().float().unsqueeze(0).to(DEVICE)

# ==============================================================================
# 4. ĐỊNH NGHĨA HÀM DỰ ĐOÁN TÙY BIẾN CHO SHAP (BẢN 6 CHANNELS / 102 DIMENSIONS)
# ==============================================================================
def custom_predict(flat_data):
    batch_size = flat_data.shape[0]
    # Khởi tạo vùng không gian Tensor 5 chiều trống cho cấu hình 6 channels
    reconstructed = np.zeros((batch_size, 6, 150, 17, 1))
    
    for i in range(batch_size):
        # Ép ma trận phẳng về dạng (6 channels x 17 khớp x 1 người) = 102 phần tử
        orig_spatial = flat_data[i].reshape(6, 17, 1)
        # Sao chép đặc trưng tĩnh này lặp lại dọc suốt chuỗi thời gian 150 khung hình
        for t in range(150):
            reconstructed[i, :, t, :, :] = orig_spatial
            
    torch_tensor = torch.tensor(reconstructed).float().to(DEVICE)
    with torch.no_grad():
        out = model(torch_tensor)
        return torch.softmax(out, dim=1).cpu().numpy()

# Chuẩn bị dữ liệu đầu vào phẳng hóa bằng cách lấy giá trị trung bình theo trục thời gian T
spatial_data = sample_tensor.mean(axis=1) # Shape: (6, 17, 1)
flat_input = spatial_data.flatten().reshape(1, -1) # Shape phẳng: (1, 102)

# ==============================================================================
# 5. THỰC THI SHAP KERNEL EXPLAINER & TÍNH GIÁ TRỊ SHAPLEY
# ==============================================================================
# Trích xuất các kích thước thực tế của mẫu dữ liệu đầu vào
current_channels = sample_tensor.shape[0]
actual_frames = sample_tensor.shape[1]  # Lấy số khung hình thực tế (ví dụ: 35)
actual_joints = sample_tensor.shape[2]  # Lấy số khớp thực tế (17)

if current_channels < 6:
    print(f"-> Phát hiện dữ liệu đầu vào chỉ có {current_channels} kênh và {actual_frames} frames.")
    print(f"-> Đang tự động tạo mảng đệm 6 kênh tương thích kích thước ({6}, {actual_frames}, {actual_joints}, 1)...")
    
    # Khởi tạo ma trận trống dựa trên đúng số khung hình thực tế của mẫu dữ liệu
    padded_tensor = np.zeros((6, actual_frames, actual_joints, 1))
    
    # Đổ các kênh thực tế hiện có (X, Y) vào mảng đệm mới
    padded_tensor[:current_channels, :, :, :] = sample_tensor
    sample_tensor = padded_tensor

# Tính toán dữ liệu không gian phẳng hóa chuẩn 102 chiều (6 channels x 17 joints)
spatial_data = sample_tensor.mean(axis=1) # Gom trục thời gian thực tế -> Shape: (6, 17, 1)
flat_input = spatial_data.flatten().reshape(1, -1) # Khóa khít cấu trúc phẳng: (1, 102)

print("-> Đang tiến hành phân tích XAI bằng thuật toán SHAP KernelExplainer...")

# Đảm bảo đầu vào là mảng NumPy thuần túy cho SHAP
if isinstance(flat_input, torch.Tensor):
    flat_input = flat_input.cpu().numpy()
else:
    flat_input = np.array(flat_input)

# Khởi tạo Explainer và tính toán Shapley với mảng không gian 102 chiều đồng bộ
explainer = shap.KernelExplainer(custom_predict, np.zeros((1, 102)))
shap_values = explainer.shap_values(flat_input)

# Bốc nhãn có xác suất dự đoán cao nhất từ mô hình
predicted_class = custom_predict(flat_input).argmax()
print(f"🎯 Mô hình nhận diện nhãn hành động: {class_names[predicted_class]}")

# Gom tổng giá trị tuyệt đối Shapley trên cả 6 kênh đầu vào cho 17 nốt xương
joint_importance = np.abs(shap_values[predicted_class].reshape(6, 17, 1)).sum(axis=0).flatten()

# ==============================================================================
# 6. TRỰC QUAN HÓA ĐỒ THỊ BIỂU DIỄN ĐỘ QUAN TRỌNG CỦA KHỚP (JOINT IMPORTANCE)
# ==============================================================================
print("-> Đang khởi tạo giao diện vẽ biểu đồ tầm quan trọng đặc trưng...")
plt.figure(figsize=(11, 7), dpi=150)

# Định nghĩa tên tiếng Anh khoa học cho 17 khớp xương theo đúng quy ước COCO
coco_joints = [
    "Nose (0)", "L-Eye (1)", "R-Eye (2)", "L-Ear (3)", "R-Ear (4)", 
    "L-Shoulder (5)", "R-Shoulder (6)", "L-Elbow (7)", "R-Elbow (8)", 
    "L-Wrist (9)", "R-Wrist (10)", "L-Hip (11)", "R-Hip (12)", 
    "L-Knee (13)", "R-Knee (14)", "L-Ankle (15)", "R-Ankle (16)"
]

# Tạo dải màu đỏ coral làm nổi bật các khớp vận động chi trên chủ chốt
colors = ['#ff6b6b' if i in [0, 7, 8, 9, 10] else '#4dadf7' for i in range(17)]

plt.barh(coco_joints, joint_importance, color=colors, edgecolor='dimgrey', height=0.6)
plt.xlabel("SHAP Value (Absolute Importance Metric)", fontsize=11, fontweight='bold')
plt.ylabel("Standard COCO Skeletal Vertices", fontsize=11, fontweight='bold')
plt.title(f"SHAP XAI: Quantitative Feature Attribution for '{class_names[predicted_class].upper()}'", fontsize=12, fontweight='bold', pad=15)
plt.grid(axis='x', linestyle='--', alpha=0.5)
plt.gca().invert_yaxis()
plt.tight_layout()

# Tạo thư mục đầu ra và tiến hành xuất lưu file đồ thị
output_fig_path = './work_dir/dav/ctrgcn_dav/shap_joint_importance.png'
os.makedirs(os.path.dirname(output_fig_path), exist_ok=True)
plt.savefig(output_fig_path, bbox_inches='tight')
plt.show()

print(f"✅ Quá trình giải thích mô hình hoàn tất! Biểu đồ XAI được lưu tại: {output_fig_path}")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Tên 6 hành động thực tế khớp 100% với ID nhãn từ 0 đến 5
class_names = ["drinking", "play_phone", "sleeping", "talking", "watch_computer", "writing"]
num_classes = len(class_names)

# ==============================================================================
# 2. KHỞI TẠO MODEL TRỰC TIẾP & ĐỔ TRỌNG SỐ (CHECKPOINT)
# ==============================================================================
print("-> Đang nạp mô hình CTR-GCN...")

# Đã sửa đổi: Khớp hoàn toàn với cấu hình 6 channels, 17 points, 1 person của bạn
model = Model(
    num_class=num_classes,   # 6 lớp hành động
    num_point=17,            # 17 khớp xương chuẩn đồ thị COCO
    num_person=1,            # 1 người tối đa trong một khung hình
    in_channels=6,           # 6 kênh đặc trưng (X, Y, Motion/Velocity...)
    graph='graph.dav.Graph',
    graph_args={'labeling_mode': 'spatial'}
).to(DEVICE)

if os.path.exists(CHECKPOINT_PATH):
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint)
    print("✅ Nạp trọng số thành công!")
else:
    print(f"🚨 [LỖI]: Không tìm thấy file trọng số tại {CHECKPOINT_PATH}!")
    exit()

model.eval()

# ==============================================================================
# 3. NẠP TẬP DỮ LIỆU KIỂM THỬ (TEST LOAD)
# ==============================================================================
print("-> Đang nạp mẫu dữ liệu...")
dataset = Feeder(data_path=TEST_DATA_PATH, label_path=TEST_LABEL_PATH)

# SỬA DÒNG 54: Trích xuất an toàn để tránh UserWarning và lỗi ép kiểu Tensor
bg_samples = []
for i in range(10):
    sample_data = dataset[i][0]
    if isinstance(sample_data, torch.Tensor):
        bg_samples.append(sample_data.clone().detach())
    else:
        bg_samples.append(torch.tensor(sample_data))
background_data = torch.stack(bg_samples).float().to(DEVICE)

# SỬA DÒNG 57: Thêm biến ẩn '_' để hứng trọn vẹn phần tử thứ 3 (index/info) từ bộ nạp Feeder
sample_tensor, sample_label, _ = dataset[0]
# Thêm trục batch -> shape chuẩn hóa: (1, 6, 150, 17, 1)
input_tensor = torch.tensor(sample_tensor).float().unsqueeze(0).to(DEVICE)

# ==============================================================================
# 4. ĐỊNH NGHĨA HÀM DỰ ĐOÁN TÙY BIẾN CHO SHAP (CUSTOM PREDICT)
# ==============================================================================
# Kích thước hình học gốc đầu vào của mô hình: (Batch, Kênh, Khung hình, Khớp, Người)
index_shape = (1, 6, 150, 17, 1)

def custom_predict(flat_data):
    batch_size = flat_data.shape[0]
    # Khởi tạo vùng không gian Tensor 5 chiều trống để tái cấu trúc lại từ mảng phẳng
    reconstructed = np.zeros((batch_size, 6, 150, 17, 1))
    
    for i in range(batch_size):
        # Khôi phục ma trận không gian tĩnh của 6 kênh đặc trưng trên 17 khớp xương
        orig_spatial = flat_data[i].reshape(6, 17, 1)
        # Sao chép đặc trưng tĩnh này lặp lại dọc suốt chuỗi thời gian 150 khung hình
        for t in range(150):
            reconstructed[i, :, t, :, :] = orig_spatial
            
    torch_tensor = torch.tensor(reconstructed).float().to(DEVICE)
    with torch.no_grad():
        out = model(torch_tensor)
        # Trả về phân phối xác suất dưới dạng Softmax
        return torch.softmax(out, dim=1).cpu().numpy()

# Chuẩn bị dữ liệu đầu vào phẳng hóa bằng cách lấy giá trị trung bình theo trục thời gian T
spatial_data = sample_tensor.mean(axis=1) # Shape rút gọn: (6, 17, 1)
flat_input = spatial_data.flatten().reshape(1, -1) # Shape phẳng: (1, 102)

# ==============================================================================
# 5. THỰC THI SHAP KERNEL EXPLAINER & TÍNH GIÁ TRỊ SHAPLEY
# ==============================================================================
print("-> Đang tiến hành phân tích XAI bằng thuật toán SHAP KernelExplainer...")

# SỬA TẠI ĐÂY: Nếu flat_input đang là Tensor, ta ép nó về mảng NumPy thuần túy
if isinstance(flat_input, torch.Tensor):
    flat_input = flat_input.cpu().numpy()
else:
    flat_input = np.array(flat_input)

# Thiết lập nền background dữ liệu mẫu số 0 và gọi Explainer
explainer = shap.KernelExplainer(custom_predict, np.zeros((1, 34)))
shap_values = explainer.shap_values(flat_input)

# Bốc nhãn có xác suất dự đoán cao nhất từ mô hình
predicted_class = custom_predict(flat_input).argmax()
print(f"🎯 Mô hình nhận diện nhãn hành động: {class_names[predicted_class]}")

# Gom tổng giá trị tuyệt đối Shapley trên cả 6 kênh đầu vào cho từng nốt xương trong số 17 nốt
joint_importance = np.abs(shap_values[predicted_class].reshape(6, 17, 1)).sum(axis=0).flatten()

# ==============================================================================
# 6. TRỰC QUAN HÓA ĐỒ THỊ BIỂU DIỄN ĐỘ QUAN TRỌNG CỦA KHỚP (JOINT IMPORTANCE)
# ==============================================================================
print("-> Đang khởi tạo giao diện vẽ biểu đồ tầm quan trọng đặc trưng...")
plt.figure(figsize=(11, 7), dpi=150)

# Định nghĩa tên tiếng Anh khoa học cho 17 khớp xương theo đúng quy ước COCO chuẩn quốc tế
coco_joints = [
    "Nose (0)", "L-Eye (1)", "R-Eye (2)", "L-Ear (3)", "R-Ear (4)", 
    "L-Shoulder (5)", "R-Shoulder (6)", "L-Elbow (7)", "R-Elbow (8)", 
    "L-Wrist (9)", "R-Wrist (10)", "L-Hip (11)", "R-Hip (12)", 
    "L-Knee (13)", "R-Knee (14)", "L-Ankle (15)", "R-Ankle (16)"
]

# Tạo dải màu làm nổi bật các khớp vận động chi trên (Cổ tay, khuỷu tay, đầu)
colors = ['#ff6b6b' if i in [0, 7, 8, 9, 10] else '#4dadf7' for i in range(17)]

plt.barh(coco_joints, joint_importance, color=colors, edgecolor='dimgrey', height=0.6)
plt.xlabel("SHAP Value (Absolute Importance Metric)", fontsize=11, fontweight='bold')
plt.ylabel("Standard COCO Skeletal Vertices", fontsize=11, fontweight='bold')
plt.title(f"SHAP XAI: Quantitative Feature Attribution for '{class_names[predicted_class].upper()}'", fontsize=12, fontweight='bold', pad=15)
plt.grid(axis='x', linestyle='--', alpha=0.5)
plt.gca().invert_yaxis()
plt.tight_layout()

# Tạo thư mục đầu ra và tiến hành xuất lưu file đồ thị
output_fig_path = './work_dir/dav/ctrgcn_joint/shap_joint_importance.png'
os.makedirs(os.path.dirname(output_fig_path), exist_ok=True)
plt.savefig(output_fig_path, bbox_inches='tight')
plt.show()

print(f"✅ Quá trình giải thích mô hình hoàn tất! Biểu đồ XAI được lưu tại: {output_fig_path}")