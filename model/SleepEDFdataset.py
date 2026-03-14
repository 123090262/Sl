import os
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from pyedflib import EdfReader
from scipy.signal import butter, filtfilt
import mne
from mne.decoding import Scaler

# --- 统计工具 ---
def calculate_transition_matrix(all_labels):
    P = np.zeros((5, 5))
    for labels in all_labels:
        for i in range(len(labels) - 1):
            curr, nxt = labels[i], labels[i+1]
            if 0 <= curr < 5 and 0 <= nxt < 5:
                P[curr, nxt] += 1
    P_prob = P / (P.sum(axis=1, keepdims=True) + 1e-9)
    return torch.tensor(P_prob, dtype=torch.float32)

def compute_fcMatrix(X_scaled):
    C = X_scaled.shape[1]
    X_flat = X_scaled.transpose(1, 0, 2).reshape(C, -1)
    fc_matrix = np.corrcoef(X_flat)
    return np.abs(fc_matrix).astype(np.float32)

# --- 核心加载与预处理 (增加 N1 重叠滑窗) ---
def load_edf_subject(psg_path, hyp_path, channels, n1_aug_stride=15):
    """
    n1_aug_stride: N1 阶段的滑动步长（秒）。若为 5，则每个 30s 的 N1 会生成 30/5 = 6 个样本。
    """
    try:
        psg = EdfReader(psg_path)
        all_ch = psg.getSignalLabels()
        fs = int(psg.getSampleFrequency(0))
        indices = [all_ch.index(ch) for ch in channels]
        signals = np.vstack([psg.readSignal(i) for i in indices])
        psg._close()
        X_raw = signals
        
        hyp = EdfReader(hyp_path)
        ann = hyp.readAnnotations()
        hyp._close()
        
        epoch_samples = 30 * fs
        num_epochs = X_raw.shape[1] // epoch_samples
        labels = np.full((num_epochs,), -1, dtype=int)
        
        label_map = {'Sleep stage W': 0, 'Sleep stage 1': 1, 'Sleep stage 2': 2, 
                     'Sleep stage 3': 3, 'Sleep stage 4': 3, 'Sleep stage R': 4}
        
        for onset, duration, desc in zip(ann[0], ann[1], ann[2]):
            if desc in label_map:
                start_epoch = int(onset // 30)
                n_epochs = int(duration // 30)
                end_epoch = min(start_epoch + n_epochs, num_epochs)
                labels[start_epoch:end_epoch] = label_map[desc]

        # 双端 W 裁剪
        sleep_idx = np.where((labels > 0))[0]
        if len(sleep_idx) > 0:
            buffer = 60 
            start_idx = max(0, sleep_idx[0] - buffer)
            end_idx = min(num_epochs, sleep_idx[-1] + buffer + 1)
            X_raw = X_raw[:, start_idx * epoch_samples : end_idx * epoch_samples]
            labels = labels[start_idx : end_idx]

        # 记录原始统计量（剔除 Unknown 后的纯净样本）
        valid_mask = (labels != -1)
        original_y = labels[valid_mask]
        
        # 执行增强切片
        X_final = []
        y_final = []
        
        current_valid_indices = np.where(valid_mask)[0]
        for idx in current_valid_indices:
            label = labels[idx]
            start_s = idx * epoch_samples
            
            # 基础样本
            X_final.append(X_raw[:, start_s : start_s + epoch_samples])
            y_final.append(label)
            
            # N1 增强：如果当前是 N1，且不是最后一个 epoch（防止越界）
            if label == 1 and (start_s + epoch_samples + fs * n1_aug_stride <= X_raw.shape[1]):
                # 以 n1_aug_stride 为步长进行重叠采样
                # 例如 30s 窗口，5s 步长，增加 5 个重叠样本
                for offset_sec in range(n1_aug_stride, 30, n1_aug_stride):
                    offset_tick = offset_sec * fs
                    if start_s + offset_tick + epoch_samples <= X_raw.shape[1]:
                        X_final.append(X_raw[:, start_s + offset_tick : start_s + offset_tick + epoch_samples])
                        y_final.append(1)

        return np.array(X_final).astype(np.float32), np.array(y_final), original_y, fs
    
    except Exception as e:
        print(f"Error loading {psg_path}: {e}")
        return None, None, None, None

def apply_custom_filter(X, fs):
    nyq = 0.5 * fs
    b1, a1 = butter(4, [0.5/nyq, 35.0/nyq], btype='band')
    X[:, 0:2, :] = filtfilt(b1, a1, X[:, 0:2, :], axis=-1)
    b2, a2 = butter(4, [0.3/nyq, 10.0/nyq], btype='band')
    X[:, 2, :] = filtfilt(b2, a2, X[:, 2, :], axis=-1)
    return X

def apply_subject_scaler_mne(X_sub, fs, channels):
    ch_types = ["eeg", "eeg", "eog"]
    info = mne.create_info(ch_names=channels, sfreq=fs, ch_types=ch_types)
    scaler = Scaler(info=info, scalings="mean")
    return scaler.fit_transform(X_sub).astype(np.float32)
    
def get_data_dict(root, sub_list, channels):
    data_dict = {}
    all_training_labels = [] 
    original_labels_all = [] # 用于最终统计原有样本量
    all_files = os.listdir(root)

    for sub_id in sub_list:
        psg_files = sorted([f for f in all_files if f.startswith(sub_id) and f.endswith("PSG.edf")])
        sub_raw_X, sub_raw_y, sub_fs = [], [], None
        
        for psg_f in psg_files:
            file_prefix = psg_f[:6]                    
            hyp_f = [f for f in all_files if f.startswith(file_prefix) and f.endswith("Hypnogram.edf")]
            if hyp_f:
                X, y, orig_y, fs = load_edf_subject(os.path.join(root, psg_f), os.path.join(root, hyp_f[0]), channels)
                if X is not None:
                    sub_raw_X.append(X)
                    sub_raw_y.append(y)
                    original_labels_all.append(orig_y) # 记录原始标签
                    sub_fs = fs
        
        if sub_raw_X:
            X_combined = np.concatenate(sub_raw_X, axis=0)
            Y_combined = np.concatenate(sub_raw_y, axis=0)
            X_filtered = apply_custom_filter(X_combined, sub_fs)
            X_scaled = apply_subject_scaler_mne(X_filtered, sub_fs, channels)
            A_fc = compute_fcMatrix(X_scaled)
            
            data_dict[sub_id] = (X_scaled, Y_combined, A_fc)
            all_training_labels.append(Y_combined)
            print(f"Loaded {sub_id}, original epoch: {len(np.concatenate(original_labels_all[-len(sub_raw_X):]))}, augmented total: {len(Y_combined)}")

    P_matrix = calculate_transition_matrix(all_training_labels)
    # 将汇总的原始标签合并为一个数组返回，用于最后的统计
    y_orig_total = np.concatenate(original_labels_all, axis=0) if original_labels_all else np.array([])
    return data_dict, P_matrix, y_orig_total

# --- Dataset 类保持不变 ---
class SeqSleepDataset(Dataset):
    def __init__(self, data_dict, sub_ids, seq_len=5):
        self.X = np.concatenate([data_dict[i][0] for i in sub_ids], axis=0)
        self.y = np.concatenate([data_dict[i][1] for i in sub_ids], axis=0)
        self.A = np.concatenate([np.repeat(data_dict[i][2][np.newaxis,...], len(data_dict[i][1]), axis=0) for i in sub_ids], axis=0)
        self.seq_len = seq_len
        self.pad = seq_len // 2
        self.X_pad = np.pad(self.X, ((self.pad, self.pad), (0,0), (0,0)), mode='constant')
        self.A_pad = np.pad(self.A, ((self.pad, self.pad), (0,0), (0,0)), mode='constant')

    def __len__(self): return len(self.y)
    def __getitem__(self, i):
        return (torch.from_numpy(self.X_pad[i:i+self.seq_len]), 
                torch.tensor(self.y[i], dtype=torch.long), 
                torch.from_numpy(self.A_pad[i:i+self.seq_len]))

if __name__ == "__main__":
    root = r"E:\EEG\dataset\sleep-edf-database-expanded-1.0.0\SleepEDF-20" 
    stage_names = ["W", "N1", "N2", "N3", "REM"]
    sub_list = sorted({f[:5] for f in os.listdir(root) if "PSG.edf" in f})[:20]

    # y_orig_total 返回的是未经过重叠滑窗增强的原始样本统计
    data_dict, P_matrix, y_orig_total = get_data_dict(root, sub_list, channels=["EEG Fpz-Cz", "EEG Pz-Oz", "EOG horizontal"])

    print("\n--- 原始样本数统计 (未增强前) ---")
    for i, name in enumerate(stage_names):
        print(f"{name}: {int((y_orig_total == i).sum())}")
    print(f"Total Original: {len(y_orig_total)}")

    print("\n--- 训练/测试实际使用的样本数 (含 N1 增强) ---")
    y_aug_total = np.concatenate([v[1] for v in data_dict.values()], axis=0)
    for i, name in enumerate(stage_names):
        print(f"{name}: {int((y_aug_total == i).sum())}")
    print(f"Total Augmented: {len(y_aug_total)}")

    print("\n转移概率矩阵 P (基于增强后的序列计算，以匹配训练逻辑)")
    P = P_matrix.cpu().numpy()
    print("      " + "  ".join([f"{n:>6s}" for n in stage_names]))
    for i, row_name in enumerate(stage_names):
        print(f"{row_name:>4s}  " + "  ".join([f'{P[i, j]:6.4f}' for j in range(5)]))