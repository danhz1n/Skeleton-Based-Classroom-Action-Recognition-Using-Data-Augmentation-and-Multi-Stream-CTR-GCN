import numpy as np
import pickle

from torch.utils.data import Dataset

from feeders import tools


class Feeder(Dataset):
    def __init__(self, data_path, label_path=None, p_interval=1, split='train', random_choose=False, random_shift=False,
                 random_move=False, random_rot=False, window_size=-1, normalization=False, debug=False, use_mmap=False,
                 bone=False, vel=False):
        """
        :param data_path: Đường dẫn file .npy
        :param label_path: Đường dẫn file .pkl
        :param split: 'train' hoặc 'test'/'val'
        :param p_interval: Khoảng tỉ lệ cắt ngẫu nhiên khi train (ví dụ: [0.5, 1])
        :param window_size: Độ dài chuỗi thời gian đích (ví dụ: 64)
        :param bone: Sử dụng dữ liệu Xương (True/False)
        :param vel: Sử dụng dữ liệu Vận tốc (True/False)
        """
        self.debug = debug
        self.data_path = data_path
        self.label_path = label_path
        self.split = split
        self.random_choose = random_choose
        self.random_shift = random_shift
        self.random_move = random_move
        self.window_size = window_size
        self.normalization = normalization
        self.use_mmap = use_mmap
        self.p_interval = p_interval
        self.random_rot = random_rot
        self.bone = bone
        self.vel = vel
        self.load_data()
        if normalization:
            self.get_mean_map()

    def load_data(self):
        # Đọc file label (.pkl)
        try:
            with open(self.label_path, 'rb') as f:
                self.sample_name, self.label = pickle.load(f)
        except:
            # Hỗ trợ giải mã nếu file pickle được tạo từ python2
            with open(self.label_path, 'rb') as f:
                self.sample_name, self.label = pickle.load(f, encoding='latin1')

        # Đọc file dữ liệu tọa độ (.npy)
        if self.use_mmap:
            self.data = np.load(self.data_path, mmap_mode='r')
        else:
            self.data = np.load(self.data_path)

    def get_mean_map(self):
        data = self.data
        N, C, T, V, M = data.shape
        self.mean_map = data.mean(axis=2, keepdims=True).mean(axis=4, keepdims=True).mean(axis=0)
        self.std_map = data.transpose((0, 2, 4, 1, 3)).reshape((N * T * M, C * V)).std(axis=0).reshape((C, 1, V, 1))

    def __len__(self):
        return len(self.label)

    def __iter__(self):
        return self

    def __getitem__(self, index):
        data_numpy = self.data[index]
        label = self.label[index]
        data_numpy = np.array(data_numpy)
        
        # 1. SỬA LỖI UNPACK (Xử lý dữ liệu 3D thiếu chiều M của bộ data_cobot_clr_zoom)
        if len(data_numpy.shape) == 3:
            # Nếu data thô có dạng (T, V, C) với C=3 ở cuối, ta chuyển về (C, T, V)
            if data_numpy.shape[2] == 3:
                data_numpy = data_numpy.transpose(2, 0, 1)
            # Chuyển từ (C, T, V) thành (C, T, V, 1) để tương thích cấu trúc mạng gốc
            data_numpy = np.expand_dims(data_numpy, axis=-1)

        # Tính toán số lượng frame thực tế có chứa dữ liệu (bỏ qua các frame rác trống)
        valid_frame_num = np.sum(data_numpy.sum(0).sum(-1).sum(-1) != 0)
        
        # Áp dụng Augmentation xoay khung xương nếu được bật
        if self.random_rot:
            data_numpy = tools.random_rot(data_numpy)
            
        # 2. CHIẾN LƯỢC BỎ RESIZE: Thay thế hoàn toàn valid_crop_resize cũ
        if self.window_size != -1:
            if valid_frame_num == 0:
                # Nếu chuỗi trống hoàn toàn, khởi tạo chuỗi zero theo window_size cố định
                data_numpy = np.zeros((data_numpy.shape[0], self.window_size, data_numpy.shape[2], data_numpy.shape[3]))
            else:
                # Cắt bỏ các frame trống ở đuôi dữ liệu thô, chỉ giữ lại các frame thực tế
                data_numpy = data_numpy[:, :valid_frame_num, :, :]
                
                C, T, V, M = data_numpy.shape
                
                if T == self.window_size:
                    pass
                elif T < self.window_size:
                    # KỸ THUẬT ZERO-PADDING: Chuỗi ngắn hơn window_size (64) -> Bù thêm frame 0 vào phía sau
                    pad_length = self.window_size - T
                    pad_tensor = np.zeros((C, pad_length, V, M), dtype=data_numpy.dtype)
                    data_numpy = np.concatenate((data_numpy, pad_tensor), axis=1)
                else:
                    # KỸ THUẬT CROPPING: Chuỗi dài hơn window_size (64) -> Tiến hành cắt phân đoạn
                    if self.split == 'train':
                        # Nếu là tập Train: Chọn vị trí bắt đầu cắt ngẫu nhiên (Random Crop) để tăng cường dữ liệu
                        p_interval = self.p_interval
                        ratio = np.random.uniform(p_interval[0], p_interval[1])
                        start_frame = int((T - self.window_size) * ratio)
                        start_frame = max(0, min(start_frame, T - self.window_size))
                        data_numpy = data_numpy[:, start_frame:start_frame + self.window_size, :, :]
                    else:
                        # Nếu là tập Test/Val: Cắt lấy phân đoạn chính giữa (Center Crop) để đảm bảo tính khách quan
                        start_frame = (T - self.window_size) // 2
                        data_numpy = data_numpy[:, start_frame:start_frame + self.window_size, :, :]

        elif self.random_choose:
            data_numpy = tools.random_choose(data_numpy, self.window_size)
        elif self.random_shift:
            data_numpy = tools.random_shift(data_numpy)
        elif self.random_move:
            data_numpy = tools.random_move(data_numpy)
            
        # Tính toán dòng dữ liệu Xương (Bone) nếu cấu hình yêu cầu
        if self.bone:
            from .bone_pairs import ntu_pairs
            bone_data_numpy = np.zeros_like(data_numpy)
            for v1, v2 in ntu_pairs:
                bone_data_numpy[:, :, v1 - 1] = data_numpy[:, :, v1 - 1] - data_numpy[:, :, v2 - 1]
            data_numpy = bone_data_numpy
            
        # Tính toán dòng dữ liệu Chuyển động (Velocity) nếu cấu hình yêu cầu
        if self.vel:
            data_numpy[:, :-1] = data_numpy[:, 1:] - data_numpy[:, :-1]
            data_numpy[:, -1] = 0

        return data_numpy, label, index

    def top_k(self, score, top_k):
        rank = score.argsort()
        hit_top_k = [l in rank[i, -top_k:] for i, l in enumerate(self.label)]
        return sum(hit_top_k) * 1.0 / len(hit_top_k)


def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod