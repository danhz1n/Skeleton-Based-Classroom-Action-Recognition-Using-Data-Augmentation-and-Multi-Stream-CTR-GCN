import os
import sys
import glob
import torch
import torch.nn as nn
import numpy as np
import pickle
from torch.utils.data import DataLoader
from tqdm import tqdm

try:
    from sklearn.manifold import TSNE
    from sklearn.neighbors import NearestNeighbors
    import matplotlib.pyplot as plt
    import seaborn as sns
except ImportError:
    print("Vui lòng cài đặt thư viện: pip install scikit-learn matplotlib seaborn tqdm")
    sys.exit(1)

from model.ctrgcn import Model
from feeders.feeder_dav import Feeder

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

        print("Đang nạp trọng số mạng Ensemble...")
        self.model_j.load_state_dict(torch.load(get_best_model(joint_dir), map_location='cpu'))
        self.model_b.load_state_dict(torch.load(get_best_model(bone_dir), map_location='cpu'))
        self.model_jm.load_state_dict(torch.load(get_best_model(jm_dir), map_location='cpu'))
        self.model_bm.load_state_dict(torch.load(get_best_model(bm_dir), map_location='cpu'))
        print("Nạp trọng số thành công!")

    def forward(self, x):
        # x is (N, 2, T, 17, 1)
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

# Bộ trích xuất đặc trưng (Feature Extractor)
class FeatureExtractor:
    def __init__(self):
        self.j_feats = []
        self.b_feats = []
        self.jm_feats = []
        self.bm_feats = []
        
    def hook_j(self, module, inp, out):
        self.j_feats.append(out.detach().cpu().numpy())
    def hook_b(self, module, inp, out):
        self.b_feats.append(out.detach().cpu().numpy())
    def hook_jm(self, module, inp, out):
        self.jm_feats.append(out.detach().cpu().numpy())
    def hook_bm(self, module, inp, out):
        self.bm_feats.append(out.detach().cpu().numpy())

def main():
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
        print(f"LỖI: {e}")
        return

    model.to(device)
    model.eval()

    # Đăng ký Hook để moi đặc trưng (256 chiều) ngay trước lớp cuối cùng (self.fc)
    # Lớp drop_out nằm ngay trước self.fc
    extractor = FeatureExtractor()
    model.model_j.drop_out.register_forward_hook(extractor.hook_j)
    model.model_b.drop_out.register_forward_hook(extractor.hook_b)
    model.model_jm.drop_out.register_forward_hook(extractor.hook_jm)
    model.model_bm.drop_out.register_forward_hook(extractor.hook_bm)

    # Khởi tạo Feeder để load toàn bộ tập Test
    print("Đang nạp tập Test...")
    dataset = Feeder(data_path='./data/dav/test_data.pkl', 
                     label_path='./data/dav/test_label.pkl',
                     bone=False, vel=False, early_fusion=False, window_size=50)
                     
    data_loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=2)

    all_labels = []
    all_preds = []
    
    print("Đang chạy mô hình để trích xuất đặc trưng (Feature Extraction)...")
    with torch.no_grad():
        for data, label, index in tqdm(data_loader):
            # Cần unsqueeze chiều M=1 để phù hợp với input của model (N, C, T, V, M)
            if len(data.shape) == 4:
                data = data.unsqueeze(-1)
            
            data = data.to(device)
            
            # Đẩy qua mạng
            output = model(data)
            pred = output.argmax(dim=1)
            
            all_labels.extend(label.numpy())
            all_preds.extend(pred.cpu().numpy())

    # Gộp các đặc trưng từ 4 mạng thành một Siêu Đặc Trưng (Super Feature) đại diện cho Ensemble
    # Mỗi mạng trả về 256 chiều -> Tổng cộng 1024 chiều
    feat_j = np.concatenate(extractor.j_feats, axis=0)
    feat_b = np.concatenate(extractor.b_feats, axis=0)
    feat_jm = np.concatenate(extractor.jm_feats, axis=0)
    feat_bm = np.concatenate(extractor.bm_feats, axis=0)
    
    ensemble_features = np.concatenate([feat_j, feat_b, feat_jm, feat_bm], axis=1) # (N, 1024)
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    
    print(f"Kích thước đặc trưng trích xuất: {ensemble_features.shape}")
    
    # ---------------------------------------------------------
    # CHẠY t-SNE
    # ---------------------------------------------------------
    print("Đang chạy t-SNE (Sẽ mất khoảng 1-2 phút)...")
    tsne = TSNE(n_components=2, perplexity=30, random_state=42, init='pca', learning_rate='auto')
    features_2d = tsne.fit_transform(ensemble_features)
    
    # ---------------------------------------------------------
    # VẼ BIỂU ĐỒ T-SNE
    # ---------------------------------------------------------
    # Các lớp hành động giả định của DAV (Sửa lại cho đúng tên thật của bạn nếu cần)
    class_names = ["drinking",
"play_phone",
"sleeping",
"talking",
"watch_computer",
"writing",
"lecture"]
    
    plt.figure(figsize=(10, 8))
    sns.scatterplot(
        x=features_2d[:, 0], y=features_2d[:, 1],
        hue=[class_names[l] for l in all_labels],
        palette=sns.color_palette("hls", 7),
        s=50, alpha=0.8
    )
    plt.title("Clustering results using the t-SNE algorithm (Ensemble 4-Streams)")
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.tight_layout()
    plt.savefig('tsne_clustering.png', dpi=300)
    print("Đã lưu biểu đồ t-SNE ra file: tsne_clustering.png")
    
    # ---------------------------------------------------------
    # THUẬT TOÁN TÌM OUTLIER (KẺ NGOẠI ĐẠO)
    # ---------------------------------------------------------
    print("\n--- BẮT ĐẦU SĂN OUTLIER ---")
    # Thuật toán: Tìm một điểm mà 5 hàng xóm gần nhất của nó trên t-SNE đều thuộc class khác
    nbrs = NearestNeighbors(n_neighbors=6, algorithm='ball_tree').fit(features_2d)
    distances, indices = nbrs.kneighbors(features_2d)
    
    outlier_idx = -1
    for i in range(len(features_2d)):
        true_class = all_labels[i]
        # 5 hàng xóm (bỏ qua chính nó ở vị trí 0)
        neighbors_classes = all_labels[indices[i][1:]]
        
        # Nếu tất cả hàng xóm đều khác class với nó -> Rất lọt thỏm (Outlier cực mạnh)
        if np.all(neighbors_classes != true_class):
            outlier_idx = i
            break
            
    if outlier_idx != -1:
        print(f"Đã tìm thấy 1 Outlier siêu rõ nét tại Video số (index): {outlier_idx}")
        print(f"=> Class thực tế: {class_names[all_labels[outlier_idx]]}")
        print(f"=> Class dự đoán sai của AI: {class_names[all_preds[outlier_idx]]}")
        
        # In ra hàng xóm
        neighbor_names = [class_names[l] for l in all_labels[indices[outlier_idx][1:]]]
        from collections import Counter
        most_common_neighbor = Counter(neighbor_names).most_common(1)[0][0]
        print(f"=> Phân tích: Video số {outlier_idx} lẽ ra là hành động '{class_names[all_labels[outlier_idx]]}', nhưng nó lại bị xếp lọt thỏm vào giữa cụm của hành động '{most_common_neighbor}'. Bạn có thể lấy video {outlier_idx} ra để minh họa trong báo cáo (giống Fig 8).")
    else:
        print("Mô hình của bạn phân cụm quá tốt, không tìm thấy Outlier cực đoan nào!")

if __name__ == '__main__':
    main()
