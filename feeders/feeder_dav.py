import pickle
import torch
import numpy as np
import random
from torch.utils.data import Dataset

class Feeder(Dataset):
    def __init__(self, data_path, label_path, random_choose=False, random_move=False, random_shift=False, random_rot=False, window_size=-1, bone=False, vel=False, early_fusion=False):
        """
        Custom Feeder cho mô hình CTR-GCN đọc dữ liệu từ file pkl tổng hợp.
        """
        self.data_path = data_path
        self.label_path = label_path
        self.random_choose = random_choose
        self.random_move = random_move
        self.random_shift = random_shift
        self.random_rot = random_rot
        self.window_size = window_size
        self.bone = bone
        self.vel = vel
        self.early_fusion = early_fusion
        
        self.load_data()

    def load_data(self):
        # Đọc dữ liệu lớn đã gom cụm
        with open(self.data_path, 'rb') as f:
            self.data = pickle.load(f)  # Kỳ vọng kích thước: (N, C, T, V, M)
        with open(self.label_path, 'rb') as f:
            self.label = pickle.load(f) # Kỳ vọng kích thước: (N,)
            
        self.sample_name = [f"sample_{i}" for i in range(len(self.label))]

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        # Lấy ra mẫu thứ idx: data_each có dạng (C, T, V, M)
        data_each = np.array(self.data[idx])
        label_each = self.label[idx]

        # Lọc bỏ các frame 0 (nếu có padding gốc)
        valid_frame_num = np.sum(data_each.sum(0).sum(-1).sum(-1) != 0)
        if valid_frame_num == 0:
            valid_frame_num = data_each.shape[1]
            
        data_each = data_each[:, :valid_frame_num, :, :]
        C, T, V, M = data_each.shape

        if self.window_size > 0:
            if T == self.window_size:
                pass
            elif T < self.window_size:
                # Bù thêm zero vào cuối nếu thiếu frame
                pad_width = self.window_size - T
                pad_array = np.zeros((C, pad_width, V, M))
                data_each = np.concatenate((data_each, pad_array), axis=1)
            else:
                if self.random_choose:
                    # Lấy một khoảng 50 frame ngẫu nhiên (dành cho tập train -> Data Augmentation)
                    start_frame = random.randint(0, T - self.window_size)
                    data_each = data_each[:, start_frame:start_frame + self.window_size, :, :]
                    
                    # Thêm Augmentation Xoay 2D ngẫu nhiên (chống Overfitting cực tốt)
                    # Xoay ngẫu nhiên từ -15 đến +15 độ (-0.26 đến 0.26 radian)
                    theta = random.uniform(-0.26, 0.26)
                    cos_t, sin_t = np.cos(theta), np.sin(theta)
                    R = np.array([[cos_t, -sin_t], [sin_t, cos_t]])
                    
                    # Chỉ xoay 2 trục X, Y
                    xy = data_each[:2].reshape(2, -1)
                    xy_rot = np.dot(R, xy)
                    data_each[:2] = xy_rot.reshape(2, self.window_size, V, M)
                else:
                    # Lấy 50 frame ở chính giữa (dành cho tập test -> Đảm bảo lấy được phần lõi của hành động)
                    start_frame = (T - self.window_size) // 2
                    data_each = data_each[:, start_frame:start_frame + self.window_size, :, :]

        # Chuẩn hóa gốc tọa độ (Centering): Dời toàn bộ khung xương về tọa độ tương đối so với mũi (khớp 0)
        # Giúp mô hình không bị nhiễu bởi việc người đứng ở góc trái hay phải màn hình
        root_joint = data_each[:, :, 0:1, :]
        data_each = data_each - root_joint

        # Tính toán các luồng dữ liệu
        dav_pairs = [
            (1, 0), (2, 0), (3, 1), (4, 2),
            (6, 5), (7, 5), (9, 7), (8, 6), (10, 8),
            (11, 5), (12, 6), (12, 11),
            (13, 11), (15, 13), (14, 12), (16, 14)
        ]

        if self.early_fusion:
            features = [data_each]
            if self.bone:
                bone_data = np.zeros_like(data_each)
                for v1, v2 in dav_pairs:
                    bone_data[:, :, v1, :] = data_each[:, :, v1, :] - data_each[:, :, v2, :]
                features.append(bone_data)
            if self.vel:
                vel_data = np.zeros_like(data_each)
                vel_data[:, :-1, :, :] = data_each[:, 1:, :, :] - data_each[:, :-1, :, :]
                features.append(vel_data)
            
            # Gộp tất cả dọc theo chiều Channels (C) -> axis=0
            data_each = np.concatenate(features, axis=0)
        else:
            # Xử lý luồng Bone (khoảng cách và hướng giữa các cặp khớp)
            if self.bone:
                bone_data_numpy = np.zeros_like(data_each)
                for v1, v2 in dav_pairs:
                    bone_data_numpy[:, :, v1, :] = data_each[:, :, v1, :] - data_each[:, :, v2, :]
                data_each = bone_data_numpy

            # Xử lý luồng Velocity (vận tốc chuyển động qua từng frame)
            if self.vel:
                data_each[:, :-1, :, :] = data_each[:, 1:, :, :] - data_each[:, :-1, :, :]
                data_each[:, -1, :, :] = 0

        # Trả về tensor dữ liệu float32 và nhãn số nguyên long
        return torch.tensor(data_each, dtype=torch.float32), torch.tensor(label_each, dtype=torch.long), idx

    def top_k(self, score, top_k):
        rank = score.argsort()
        hit_top_k = [l in rank[i, -top_k:] for i, l in enumerate(self.label)]
        return sum(hit_top_k) * 1.0 / len(hit_top_k)