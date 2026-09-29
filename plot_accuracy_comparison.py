import pickle
import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns

# Tên các hành động
classes = ["drinking", "play_phone", "sleeping", "talking", "watch_computer", "writing", "lecture"]

# Hàm tính accuracy per class
def get_class_accuracy(label_path, dirs, alphas):
    with open(label_path, 'rb') as f:
        labels = pickle.load(f)
        
    results = []
    rates = []
    
    for d, alpha in zip(dirs, alphas):
        if d is not None and os.path.exists(d):
            with open(os.path.join(d, 'epoch1_test_score.pkl'), 'rb') as f:
                results.append(list(pickle.load(f).items()))
            rates.append(alpha)
            
    if len(results) == 0:
        return [0]*7
        
    class_correct = {i: 0 for i in range(7)}
    class_total = {i: 0 for i in range(7)}
    
    max_len = min([len(labels)] + [len(r) for r in results])
    
    for i in range(max_len):
        l = int(labels[i])
        r = np.zeros_like(results[0][i][1])
        for idx, res in enumerate(results):
            _, scores = res[i]
            r += scores * rates[idx]
        
        pred = np.argmax(r)
        
        class_total[l] += 1
        if pred == l:
            class_correct[l] += 1
            
    # Tính phần trăm %
    class_acc = []
    for i in range(7):
        if class_total[i] > 0:
            class_acc.append((class_correct[i] / class_total[i]) * 100)
        else:
            class_acc.append(0)
            
    return class_acc

# 1. Lấy kết quả từ dav/EduAction (4 luồng)
edu_label_path = './data/dav/EduAction/test_label.pkl'
edu_dirs = [
    './work_dir/dav/EduAction/ctrgcn_rawjoint/',
    './work_dir/dav/EduAction/ctrgcn_rawbone/',
    './work_dir/dav/EduAction/ctrgcn_rawjointmotion/',
    './work_dir/dav/EduAction/ctrgcn_rawbonemotion/'
]
edu_alphas = [0.6, 0.6, 0.4, 0.4]
acc_edu = get_class_accuracy(edu_label_path, edu_dirs, edu_alphas)

# 2. Lấy kết quả từ dav (4 luồng chưa process)
dav_label_path = './data/dav/test_label.pkl'
dav_dirs = [
    './work_dir/dav/ctrgcn_joint/', 
    './work_dir/dav/ctrgcn_bone/', 
    './work_dir/dav/ctrgcn_jointmotion/', 
    './work_dir/dav/ctrgcn_bonemotion/'
]
dav_alphas = [0.6, 0.6, 0.4, 0.4]
acc_dav = get_class_accuracy(dav_label_path, dav_dirs, dav_alphas)

# 3. Vẽ biểu đồ so sánh
plt.figure(figsize=(12, 7))
sns.set_theme(style="whitegrid")

x = np.arange(len(classes))
width = 0.35

plt.bar(x - width/2, acc_edu, width, label='Unprocessed Data (Original, 4-Stream)', color='#dd8452')
plt.bar(x + width/2, acc_dav, width, label='Processed Data (After-processing, 4-Stream)', color='#4c72b0')

# Thêm số % lên trên cột
for i in range(len(classes)):
    plt.text(i - width/2, acc_edu[i] + 1, f"{acc_edu[i]:.1f}%", ha='center', va='bottom', fontsize=9, fontweight='bold', color='#dd8452')
    plt.text(i + width/2, acc_dav[i] + 1, f"{acc_dav[i]:.1f}%", ha='center', va='bottom', fontsize=9, fontweight='bold', color='#4c72b0')

plt.ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
plt.title('Per-Class Accuracy Comparison: Original vs After-processing', fontsize=16, fontweight='bold', pad=20)
plt.xticks(x, classes, rotation=45, ha='right', fontsize=11)
plt.ylim(0, 110) # Để có chỗ trống viết số %
plt.legend(fontsize=11, loc='upper right')

plt.tight_layout()
plt.savefig('accuracy_comparison.png', dpi=300)
print("Đã lưu biểu đồ vào file 'accuracy_comparison.png'")
