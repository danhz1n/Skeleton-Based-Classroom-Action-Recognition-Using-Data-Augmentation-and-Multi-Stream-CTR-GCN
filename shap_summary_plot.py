import os
import sys
import glob
import torch
import torch.nn as nn
import numpy as np
import pickle
from tqdm import tqdm
import logging

try:
    import shap
    import matplotlib.pyplot as plt
except ImportError:
    print("Vui lòng cài đặt thư viện: pip install shap matplotlib tqdm")
    sys.exit(1)

from model.ctrgcn import Model

# Cấu hình logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

class CTRGCN_EnsembleWrapper(nn.Module):
    def __init__(self, model_args):
        super().__init__()
        self.model_j = Model(**model_args)
        self.model_b = Model(**model_args)
        self.model_jm = Model(**model_args)
        self.model_bm = Model(**model_args)
        self.alpha = [0.6, 0.6, 0.4, 0.4]
        self.dav_pairs = [
            (1, 0), (2, 0), (3, 1), (4, 2),
            (6, 5), (7, 5), (9, 7), (8, 6), (10, 8),
            (11, 5), (12, 6), (12, 11),
            (13, 11), (15, 13), (14, 12), (16, 14)
        ]

    def load_weights(self, joint_dir, bone_dir, jm_dir, bm_dir):
        def get_best_model(directory):
            pt_files = glob.glob(os.path.join(directory, "*.pt"))
            if not pt_files:
                raise FileNotFoundError(f"Không tìm thấy file .pt nào trong {directory}")
            return pt_files[-1] 

        logging.info("Đang nạp trọng số mạng Ensemble...")
        self.model_j.load_state_dict(torch.load(get_best_model(joint_dir), map_location='cpu'))
        self.model_b.load_state_dict(torch.load(get_best_model(bone_dir), map_location='cpu'))
        self.model_jm.load_state_dict(torch.load(get_best_model(jm_dir), map_location='cpu'))
        self.model_bm.load_state_dict(torch.load(get_best_model(bm_dir), map_location='cpu'))
        logging.info("Nạp trọng số thành công!")

    def forward(self, x):
        out_j = self.model_j(x)
        x_b = torch.zeros_like(x)
        for v1, v2 in self.dav_pairs:
            x_b[:, :, :, v1, :] = x[:, :, :, v1, :] - x[:, :, :, v2, :]
        out_b = self.model_b(x_b)
        x_jm = torch.zeros_like(x)
        x_jm[:, :, :-1, :, :] = x[:, :, 1:, :, :] - x[:, :, :-1, :, :]
        out_jm = self.model_jm(x_jm)
        x_bm = torch.zeros_like(x_b)
        x_bm[:, :, :-1, :, :] = x_b[:, :, 1:, :, :] - x_b[:, :, :-1, :, :]
        out_bm = self.model_bm(x_bm)
        
        out_ensemble = (self.alpha[0] * out_j + 
                        self.alpha[1] * out_b + 
                        self.alpha[2] * out_jm + 
                        self.alpha[3] * out_bm)
        return out_ensemble

def get_balanced_sample_data(data_path, label_path, samples_per_class=5):
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    with open(label_path, 'rb') as f:
        labels = np.array(pickle.load(f))
        
    num_classes = 7
    selected_indices = []
    
    # Lấy đều mỗi class N mẫu
    for c in range(num_classes):
        idx_c = np.where(labels == c)[0]
        if len(idx_c) > samples_per_class:
            idx_c = np.random.choice(idx_c, samples_per_class, replace=False)
        selected_indices.extend(idx_c)
        
    selected_data = []
    window_size = 50
    
    for idx in selected_indices:
        data_sample = data[idx]
        valid_frame_num = np.sum(data_sample.sum(0).sum(-1).sum(-1) != 0)
        data_sample = data_sample[:, :valid_frame_num, :, :]
        
        T = data_sample.shape[1]
        if T > window_size:
            start_frame = (T - window_size) // 2
            data_sample = data_sample[:, start_frame:start_frame + window_size, :, :]
        elif T < window_size:
            pad_width = window_size - T
            pad_array = np.zeros((data_sample.shape[0], pad_width, data_sample.shape[2], data_sample.shape[3]))
            data_sample = np.concatenate((data_sample, pad_array), axis=1)
            
        root_joint = data_sample[:, :, 0:1, :]
        data_sample = data_sample - root_joint
        selected_data.append(data_sample)
        
    tensor_data = torch.tensor(np.array(selected_data), dtype=torch.float32)
    return tensor_data, labels[selected_indices]

def main():
    # Cài đặt tham số
    samples_per_class = 5 # Tổng cộng 35 mẫu. Có thể tăng lên nếu máy khỏe.
    
    model_args = {
        'num_class': 7,
        'num_point': 17,
        'num_person': 1,
        'in_channels': 2,
        'graph': 'graph.dav.Graph',
        'graph_args': {'labeling_mode': 'spatial'},
        'drop_out': 0.6
    }
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CTRGCN_EnsembleWrapper(model_args)
    
    try:
        model.load_weights(
            joint_dir='./work_dir/dav/ctrgcn_joint/',
            bone_dir='./work_dir/dav/ctrgcn_bone/',
            jm_dir='./work_dir/dav/ctrgcn_jointmotion/',
            bm_dir='./work_dir/dav/ctrgcn_bonemotion/'
        )
    except Exception as e:
        logging.error(f"LỖI: {e}")
        return

    model.to(device)
    model.eval()

    logging.info(f"Đang trích xuất {samples_per_class} mẫu ngẫu nhiên cho mỗi Class (Tổng {samples_per_class * 7} mẫu)...")
    sample_tensor, sample_labels = get_balanced_sample_data('./data/dav/test_data.pkl', './data/dav/test_label.pkl', samples_per_class)
    sample_tensor = sample_tensor.to(device)
    
    # Tạo Background Data (10 mẫu random)
    background_idx = np.random.choice(len(sample_tensor), min(10, len(sample_tensor)), replace=False)
    background = sample_tensor[background_idx]
    
    logging.info("Đang khởi tạo SHAP GradientExplainer...")
    explainer = shap.GradientExplainer(model, background)
    
    logging.info("Đang tính toán SHAP values (Quá trình này có thể mất 5-15 phút)...")
    shap_values = explainer.shap_values(sample_tensor)
    
    logging.info("Tính toán thành công! Đang tổng hợp dữ liệu để vẽ biểu đồ...")
    
    # shap_values là một list có 7 phần tử (tương ứng 7 classes).
    # Mỗi phần tử có kích thước: (N_samples, C, T, V, M) -> (35, 2, 50, 17, 1)
    # Ta cần gộp nó lại thành (35, 17) bằng cách tính tổng giá trị tuyệt đối trên các trục C, T, M.
    
    processed_shap_values = []
    for class_idx in range(7):
        target_shap = shap_values[class_idx]
        if torch.is_tensor(target_shap):
            target_shap = target_shap.cpu().detach().numpy()
            
        target_shap = np.squeeze(target_shap) # Bỏ các chiều kích thước 1
        
        # Kiểm tra số chiều và sum
        if target_shap.ndim == 4: # (N=35, C=2, T=50, V=17)
            # Tính tổng trị tuyệt đối theo trục kênh (1) và thời gian (2)
            agg_shap = np.sum(np.abs(target_shap), axis=(1, 2))
        else:
            # Fallback
            axes_to_sum = tuple(i for i, dim in enumerate(target_shap.shape) if i != 0 and dim != 17)
            agg_shap = np.sum(np.abs(target_shap), axis=axes_to_sum)
            
        processed_shap_values.append(agg_shap)
        
    # Tạo tên của 17 khớp (Khớp 0 -> Khớp 16)
    feature_names = [f"Joint {i}" for i in range(17)]
    
    # Bạn có thể đổi tên class_names thành tên thật của bộ dữ liệu
    class_names = [
        "Class 0", "Class 1", "Class 2", "Class 3", 
        "Class 4", "Class 5", "Class 6"
    ]
    
    # Vẽ Summary Plot dạng Bar (So sánh độ học của các class)
    plt.figure(figsize=(12, 8))
    shap.summary_plot(
        processed_shap_values, 
        features=None, 
        feature_names=feature_names, 
        class_names=class_names, 
        plot_type="bar", 
        show=False,
        max_display=17 # Hiện cả 17 khớp
    )
    plt.title("Biểu đồ SHAP: Mức độ đóng góp của từng khớp xương theo từng Hành động", fontsize=14, pad=20)
    plt.tight_layout()
    plt.savefig('shap_summary_bar.png', dpi=300)
    logging.info("HOÀN TẤT! Đã lưu biểu đồ thành công ra file: shap_summary_bar.png")

if __name__ == '__main__':
    main()
