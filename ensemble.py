import argparse
import pickle
import os

import numpy as np
from tqdm import tqdm

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset',
                        required=True,
                        choices={'ntu/xsub', 'ntu/xview', 'ntu120/xsub', 'ntu120/xset', 'NW-UCLA', 'dav', 'cobot'},
                        help='the work folder for storing results')
    parser.add_argument('--alpha',
                        default=1,
                        help='weighted summation',
                        type=float)

    parser.add_argument('--joint-dir',
                        help='Directory containing "epoch1_test_score.pkl" for joint eval results')
    parser.add_argument('--bone-dir',
                        help='Directory containing "epoch1_test_score.pkl" for bone eval results')
    parser.add_argument('--joint-motion-dir', default=None)
    parser.add_argument('--bone-motion-dir', default=None)

    arg = parser.parse_args()

    dataset = arg.dataset
    if 'UCLA' in arg.dataset:
        label = []
        with open('./data/' + 'NW-UCLA/' + '/val_label.pkl', 'rb') as f:
            data_info = pickle.load(f)
            for index in range(len(data_info)):
                info = data_info[index]
                label.append(int(info['label']) - 1)
    elif 'ntu120' in arg.dataset:
        if 'xsub' in arg.dataset:
            npz_data = np.load('./data/' + 'ntu120/' + 'NTU120_CSub.npz')
            label = np.where(npz_data['y_test'] > 0)[1]
        elif 'xset' in arg.dataset:
            npz_data = np.load('./data/' + 'ntu120/' + 'NTU120_CSet.npz')
            label = np.where(npz_data['y_test'] > 0)[1]
    elif 'ntu' in arg.dataset:
        if 'xsub' in arg.dataset:
            npz_data = np.load('./data/' + 'ntu/' + 'NTU60_CS.npz')
            label = np.where(npz_data['y_test'] > 0)[1]
        elif 'xview' in arg.dataset:
            npz_data = np.load('./data/' + 'ntu/' + 'NTU60_CV.npz')
            label = np.where(npz_data['y_test'] > 0)[1]
    elif 'dav' in arg.dataset.lower():
        with open('./data/dav/test_label.pkl', 'rb') as f:
            label = pickle.load(f)
    elif 'cobot' in arg.dataset.lower():
        with open('./data/data_cobot_clr_new/data_cobot_clr_new/xsub/val_label.pkl', 'rb') as f:
            _, label = pickle.load(f)
            label = list(label)
    else:
        raise NotImplementedError

    arg.alpha = [0.6, 0.6, 0.4, 0.4]
    rates = []
    results = []
    
    if arg.joint_dir is not None:
        with open(os.path.join(arg.joint_dir, 'epoch1_test_score.pkl'), 'rb') as f:
            results.append(list(pickle.load(f).items()))
        rates.append(arg.alpha[0])
        
    if arg.bone_dir is not None:
        with open(os.path.join(arg.bone_dir, 'epoch1_test_score.pkl'), 'rb') as f:
            results.append(list(pickle.load(f).items()))
        rates.append(arg.alpha[1])
        
    if arg.joint_motion_dir is not None:
        with open(os.path.join(arg.joint_motion_dir, 'epoch1_test_score.pkl'), 'rb') as f:
            results.append(list(pickle.load(f).items()))
        rates.append(arg.alpha[2])
        
    if arg.bone_motion_dir is not None:
        with open(os.path.join(arg.bone_motion_dir, 'epoch1_test_score.pkl'), 'rb') as f:
            results.append(list(pickle.load(f).items()))
        rates.append(arg.alpha[3])

    if len(results) == 0:
        print("Vui lòng cung cấp ít nhất 1 thư mục kết quả (VD: --joint-dir, --bone-dir...)")
        exit()

    right_num = total_num = right_num_5 = 0
    max_len = min([len(label)] + [len(r) for r in results])
    
    for i in tqdm(range(max_len)):
        l = label[i]
        
        # Gộp điểm của tất cả các luồng được cung cấp
        r = np.zeros_like(results[0][i][1])
        for idx, res in enumerate(results):
            _, scores = res[i]
            r += scores * rates[idx]
            
        rank_5 = r.argsort()[-5:]
        right_num_5 += int(int(l) in rank_5)
        r = np.argmax(r)
        right_num += int(r == int(l))
        total_num += 1

    acc = right_num / total_num
    acc5 = right_num_5 / total_num

    print('Top1 Acc: {:.4f}%'.format(acc * 100))
    print('Top5 Acc: {:.4f}%'.format(acc5 * 100))
    
    # Ghi log kết quả
    import datetime
    with open('ensemble_log.txt', 'a', encoding='utf-8') as f:
        f.write(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Dataset: {arg.dataset} (ensemble.py)\n")
        f.write(f"Alpha: {arg.alpha}\n")
        f.write(f"Top1 Acc: {acc * 100:.4f}%\n")
        f.write(f"Top5 Acc: {acc5 * 100:.4f}%\n")
        f.write("-" * 50 + "\n")
