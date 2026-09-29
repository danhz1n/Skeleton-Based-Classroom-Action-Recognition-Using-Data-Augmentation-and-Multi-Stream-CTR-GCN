"""
XAI - SHAP KernelExplainer for Ensemble
Mỗi class 1 case đại diện → Bar chart top joints quan trọng
DAV dataset: 17 joints COCO, 7 classes, Ensemble 4 streams (in_channels=2)
"""

import os
import sys
import io
import glob
import numpy as np
import torch
import torch.nn as nn
import pickle
import shap
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Cố định encoding UTF-8 cho print trên Windows
if sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# ─────────────────────────────────────────────
# 1. CONFIG
# ─────────────────────────────────────────────
DATA_PATH  = './data/dav/test_data.pkl'
LABEL_PATH = './data/dav/test_label.pkl'

JOINT_DIR = './work_dir/dav/ctrgcn_joint/'
BONE_DIR = './work_dir/dav/ctrgcn_bone/'
JM_DIR = './work_dir/dav/ctrgcn_jointmotion/'
BM_DIR = './work_dir/dav/ctrgcn_bonemotion/'

NUM_CLASSES = 7
NUM_JOINTS  = 17
WINDOW_SIZE = 50
IN_CHANNELS = 2  
TOP_K       = 10   

CLASS_NAMES = [
    "drinking",
    "play_phone",
    "sleeping",
    "talking",
    "watch_computer",
    "writing",
    "lecture"
]

JOINT_NAMES = [
    'Nose', 'L_Eye', 'R_Eye', 'L_Ear', 'R_Ear', 
    'L_Shoulder', 'R_Shoulder', 'L_Elbow', 'R_Elbow', 
    'L_Wrist', 'R_Wrist', 'L_Hip', 'R_Hip', 
    'L_Knee', 'R_Knee', 'L_Ankle', 'R_Ankle'
]

# ─────────────────────────────────────────────
# 2. ENSEMBLE WRAPPER
# ─────────────────────────────────────────────
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

# ─────────────────────────────────────────────
# 3. HELPER & CUSTOM PREDICT
# ─────────────────────────────────────────────
_current_sample_tensor = None   

def custom_predict(flat_inputs):
    """
    flat_inputs: (n_samples, 17) -> Binary mask cho 17 khớp (1 = Giữ, 0 = Xóa)
    """
    results = []
    for row in flat_inputs:
        mask = row.reshape(1, 1, 17, 1)          
        tensor = _current_sample_tensor.copy()   
        
        # Occlusion: di chuyển các khớp bị tắt về vị trí của khớp Mũi (Nose - Khớp 0)
        # Giúp loại bỏ hoàn toàn nhiễu tọa độ tuyệt đối
        nose_pos = tensor[:, :, 0:1, :] # (2, 50, 1, 1)
        tensor = tensor * mask + nose_pos * (1 - mask)

        t = torch.tensor(tensor[np.newaxis], dtype=torch.float32)
        with torch.no_grad():
            logits = model(t)
            # Chia cho 2.0 (tổng alpha) để giảm hiện tượng Softmax Sharpening
            logits = logits / 2.0
        probs = torch.softmax(logits, dim=1).numpy()[0]
        results.append(probs)
    return np.array(results)

# ─────────────────────────────────────────────
# MAIN SCRIPT
# ─────────────────────────────────────────────
def main():
    global _current_sample_tensor
    
    print("Loading data via Feeder...")
    dataset = Feeder(data_path=DATA_PATH, label_path=LABEL_PATH,
                     bone=False, vel=False, early_fusion=False, window_size=WINDOW_SIZE)
    
    print(f"  Num samples: {len(dataset)}")

    print("Loading model...")
    model_args = {
        'num_class': NUM_CLASSES,
        'num_point': NUM_JOINTS,
        'num_person': 1,
        'in_channels': IN_CHANNELS,
        'graph': 'graph.dav.Graph',
        'graph_args': {'labeling_mode': 'spatial'},
        'drop_out': 0.0
    }
    
    global model
    model = CTRGCN_EnsembleWrapper(model_args)
    try:
        model.load_weights(JOINT_DIR, BONE_DIR, JM_DIR, BM_DIR)
    except Exception as e:
        print(f"LỖI LOAD TRỌNG SỐ: {e}")
        return
    model.eval()
    print("  Model loaded OK")

    print("Selecting representative cases...")
    selected = {}   

    for idx in range(len(dataset)):
        data_np, label, _ = dataset[idx]
        if isinstance(data_np, torch.Tensor):
            data_np = data_np.detach().cpu().numpy()
            
        if len(data_np.shape) == 3:
            data_np = np.expand_dims(data_np, axis=-1) # (C, T, V, M)
            
        cls = int(label)
        if cls in selected:
            continue
            
        t = torch.tensor(data_np[np.newaxis], dtype=torch.float32)
        with torch.no_grad():
            logits = model(t)
        pred = int(logits.argmax(dim=1).item())
        if pred == cls:
            selected[cls] = idx
        if len(selected) == NUM_CLASSES:
            break

    for cls in range(NUM_CLASSES):
        if cls not in selected:
            # fallback nếu model quá kém không đoán trúng được class này
            selected[cls] = 0 # Chỉ là fallback tạm thời

    print(f"  Selected: { {k: v for k,v in sorted(selected.items())} }")

    print("Computing SHAP values...")
    all_joint_importance = {}   
    all_pred_class = {}
    all_pred_prob = {}

    for cls in range(NUM_CLASSES):
        idx = selected[cls]
        data_np, label, _ = dataset[idx]
        if isinstance(data_np, torch.Tensor):
            data_np = data_np.detach().cpu().numpy()
            
        if len(data_np.shape) == 3:
            data_np = np.expand_dims(data_np, axis=-1)
            
        _current_sample_tensor = data_np

        # flat_input: 17 khớp bật (1.0)
        flat_input = np.ones((1, NUM_JOINTS), dtype=np.float32)

        # background: 17 khớp tắt (0.0) -> Trở thành gốc tọa độ
        background = np.zeros((1, NUM_JOINTS), dtype=np.float32)

        explainer = shap.KernelExplainer(custom_predict, background)
        shap_values = explainer.shap_values(flat_input, nsamples=100)

        probs = custom_predict(flat_input)[0]
        pred_cls = int(np.argmax(probs))
        pred_prob = float(probs[pred_cls])

        all_pred_class[cls] = pred_cls
        all_pred_prob[cls]  = pred_prob

        # Trích xuất SHAP values
        if isinstance(shap_values, list):
            sv = shap_values[pred_cls]              
        else:
            if shap_values.ndim == 3:
                sv = shap_values[:, :, pred_cls]
            else:
                sv = shap_values

        joint_imp = np.abs(sv).flatten() # (17,)
        all_joint_importance[cls] = joint_imp

        print(f"  Class {cls} ({CLASS_NAMES[cls]}): "
              f"pred={CLASS_NAMES[pred_cls]} "
              f"prob={pred_prob:.2%} "
              f"top_joint={JOINT_NAMES[int(np.argmax(joint_imp))]}")

    print("Plotting...")
    COLS = 4
    ROWS = 2
    fig, axes = plt.subplots(ROWS, COLS, figsize=(20, 9))
    fig.suptitle('SHAP Joint Importance — One Representative Case per Class (Ensemble)',
                 fontsize=16, fontweight='bold', y=1.01)

    cmap = plt.get_cmap('tab10')

    for cls in range(NUM_CLASSES):
        row = cls // COLS
        col = cls  % COLS
        ax  = axes[row, col]

        joint_imp = all_joint_importance[cls]

        top_idx   = np.argsort(joint_imp)[::-1][:TOP_K]
        top_names = [JOINT_NAMES[i] for i in top_idx]
        top_vals  = joint_imp[top_idx]

        color = cmap(cls)
        bars  = ax.barh(range(TOP_K), top_vals[::-1],
                        color=color, alpha=0.85, edgecolor='white')
        ax.set_yticks(range(TOP_K))
        ax.set_yticklabels(top_names[::-1], fontsize=9)
        ax.set_xlabel('|SHAP| importance', fontsize=8)

        gt_name   = CLASS_NAMES[cls]
        pred_name = CLASS_NAMES[all_pred_class[cls]]
        prob      = all_pred_prob[cls]
        correct   = '✓' if cls == all_pred_class[cls] else '✗'
        ax.set_title(f'{correct} GT: {gt_name}\nPred: {pred_name} ({prob:.1%})',
                     fontsize=10, fontweight='bold',
                     color='green' if cls == all_pred_class[cls] else 'red')

        ax.spines[['top','right']].set_visible(False)
        ax.grid(axis='x', linestyle='--', alpha=0.4)

    if NUM_CLASSES < ROWS * COLS:
        for extra in range(NUM_CLASSES, ROWS * COLS):
            axes[extra // COLS, extra % COLS].set_visible(False)

    plt.tight_layout()
    plt.savefig('xai_shap_bar_7classes.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: xai_shap_bar_7classes.png")

    avg_imp = np.mean([all_joint_importance[c] for c in range(NUM_CLASSES)], axis=0)  
    top_idx_global   = np.argsort(avg_imp)[::-1]
    top_names_global = [JOINT_NAMES[i] for i in top_idx_global]

    fig2, ax2 = plt.subplots(figsize=(10, 5))
    colors = [cmap(i / NUM_JOINTS) for i in range(NUM_JOINTS)]
    ax2.bar(range(NUM_JOINTS), avg_imp[top_idx_global],
            color=colors, edgecolor='white', alpha=0.9)
    ax2.set_xticks(range(NUM_JOINTS))
    ax2.set_xticklabels(top_names_global, rotation=45, ha='right', fontsize=9)
    ax2.set_ylabel('Avg |SHAP| importance', fontsize=11)
    ax2.set_title('Global Joint Importance (averaged over 7 classes)',
                  fontsize=13, fontweight='bold')
    ax2.spines[['top','right']].set_visible(False)
    ax2.grid(axis='y', linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.savefig('xai_shap_global.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: xai_shap_global.png")

if __name__ == '__main__':
    main()