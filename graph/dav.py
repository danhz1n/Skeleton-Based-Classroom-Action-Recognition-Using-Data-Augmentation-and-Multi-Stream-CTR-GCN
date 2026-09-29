# graph/cobot.py
import numpy as np

# Định nghĩa các cặp nốt xương nối với nhau (Ví dụ minh họa cho 17 nốt)
# Bạn hãy thay đổi các cặp số này dựa theo đúng thứ tự khớp xương trong file .pkl của bạn
neighbor_17 = [
    (0, 1), (0, 2), (1, 3), (2, 4),           # Khớp vùng mặt/đầu
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # Thân trên và 2 tay (Rất quan trọng cho hành động drinking)
    (5, 11), (6, 12), (11, 12),               # Thân giữa / Hông
    (11, 13), (13, 15), (12, 14), (14, 16)    # Hai chân
]

class Graph:
    def __init__(self, labeling_mode='spatial'):
        self.num_node = 17
        self.self_link = [(i, i) for i in range(self.num_node)]
        self.neighbor = neighbor_17
        self.edge = self.self_link + self.neighbor
        
        # Khởi tạo ma trận kề (A)
        self.A = self.get_adjacency_matrix(labeling_mode)

    def get_adjacency_matrix(self, labeling_mode):
        if labeling_mode == 'spatial':
            A = []
            # CTR-GCN chia ma trận kề thành 3 phân nhóm: Self-link, Đi vào (Inward), Đi ra (Outward)
            A_self = np.eye(self.num_node)
            A_inward = np.zeros((self.num_node, self.num_node))
            A_outward = np.zeros((self.num_node, self.num_node))
            
            for i, j in self.neighbor:
                A_inward[j, i] = 1
                A_outward[i, j] = 1
                
            A = np.stack((A_self, A_inward, A_outward))
            return A
        else:
            raise ValueError(f"Do không hỗ trợ chế độ labeling: {labeling_mode}")

if __name__ == '__main__':
    # Chạy thử để kiểm tra cấu trúc ma trận kề
    g = Graph()
    print("Khởi tạo Graph thành công! Kích thước ma trận kề:", g.A.shape)