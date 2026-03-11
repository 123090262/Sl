import os
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, Dataset
from pyedflib import EdfReader
from scipy.signal import butter, filtfilt

# --- 1. 单个受试者/记录读取与预处理 ---

def load_edf_record(psg_path, hyp_path, channels=["EEG Fpz-Cz", "EEG Pz-Oz"]):
    """
    读取单条记录，执行双端 W 裁剪，并剔除 Unknown/无效标签
    """
    try:
        psg_edf = EdfReader(psg_path)
        all_labels = psg_edf.getSignalLabels()
        fs = int(psg_edf.getSampleFrequency(0))
        idx = [all_labels.index(ch) for ch in channels]
        signals = [psg_edf.readSignal(i) for i in idx]
        psg_edf._close()
        X = np.vstack(signals) 
    except Exception as e:
        print(f"读取信号出错 {psg_path}: {e}")
        return None, None, None

    try:
        hyp_edf = EdfReader(hyp_path)
        annotations = hyp_edf.readAnnotations()
        hyp_edf._close()
        
        epoch_len = 30 * fs
        num_epochs = X.shape[1] // epoch_len
        # 初始化为 -1，意味着所有未在 label_map 中定义的（如 '?', 'Movement time'）都会被标记为无效
        labels = np.full((num_epochs,), -1, dtype=int)
        
        label_map = {
            'Sleep stage W': 0,
            'Sleep stage 1': 1,
            'Sleep stage 2': 2,
            'Sleep stage 3': 3,
            'Sleep stage 4': 3,
            'Sleep stage R': 4
        }
        
        for onset, duration, desc in zip(annotations[0], annotations[1], annotations[2]):
            if desc in label_map:
                start_epoch = int(onset // 30)
                n_epochs = int(duration // 30)
                end_epoch = min(start_epoch + n_epochs, num_epochs)
                labels[start_epoch:end_epoch] = label_map[desc]

        # --- 双端 W 裁剪逻辑 ---
        # 寻找真正的睡眠区间（1, 2, 3, 4）
        sleep_indices = np.where((labels > 0))[0]
        
        if len(sleep_indices) > 0:
            first_sleep = sleep_indices[0]
            last_sleep = sleep_indices[-1]
            buffer = 60 # 30分钟
            
            start_idx = max(0, first_sleep - buffer)
            end_idx = min(num_epochs, last_sleep + buffer + 1)
            
            X = X[:, start_idx * epoch_len : end_idx * epoch_len]
            labels = labels[start_idx : end_idx]
        
        # --- 剔除 Unknown 和无效标签 ---
        valid_idx = np.where(labels != -1)[0]
        X_final = []
        for i in valid_idx:
            X_final.append(X[:, i*epoch_len : (i+1)*epoch_len])
            
        return np.array(X_final).astype(np.float32), labels[valid_idx], fs
        
    except Exception as e:
        print(f"解析标签出错 {hyp_path}: {e}")
        return None, None, None
def sleep_bandpass(data, fs, lowcut=0.3, highcut=35.0, order=4):
    """带通滤波 (同 ISRUC)"""
    nyq = 0.5 * fs
    b, a = butter(order, [lowcut/nyq, highcut/nyq], btype='band')
    return filtfilt(b, a, data, axis=-1)

def apply_scaler(X):
    """按通道标准化 (同 ISRUC)"""
    for c in range(X.shape[1]):
        mean, std = X[:, c, :].mean(), X[:, c, :].std()
        X[:, c, :] = (X[:, c, :] - mean) / (std + 1e-8)
    return X

# --- 2. 数据组织函数 ---

def get_subject_data_dict(root_path, subject_list, channels):
    """
    更新后的逻辑：循环加载每个受试者的所有晚间数据 (Night 1 & Night 2)
    """
    data_dict = {}
    all_files = os.listdir(root_path)
    
    for sub_id in subject_list:
        # 匹配该受试者所有的 PSG 文件（如 SC4001, SC4002 等）
        psg_files = sorted([f for f in all_files if f.startswith(sub_id) and f.endswith("PSG.edf")])
        
        all_x_sub, all_y_sub = [], []
        
        for psg_f in psg_files:
            # 提取文件前缀以匹配对应的 Hypnogram
            # 例如从 SC4001E0-PSG.edf 提取出 SC4001
            file_prefix = psg_f[:6]
            hyp_f = [f for f in all_files if f.startswith(file_prefix) and f.endswith("Hypnogram.edf")]
            
            if len(hyp_f) > 0:
                psg_path = os.path.join(root_path, psg_f)
                hyp_path = os.path.join(root_path, hyp_f[0])
                
                # 调用带裁剪逻辑的 load_edf_record
                X, y, fs = load_edf_record(psg_path, hyp_path, channels=channels)
                
                if X is not None:
                    # 预处理
                    X = sleep_bandpass(X, fs)
                    X = apply_scaler(X)
                    
                    all_x_sub.append(X.astype(np.float32))
                    all_y_sub.append(y)
            else:
                print(f"Warning: No hypnogram found for {psg_f}")

        # 将该受试者的多晚数据合并为一个条目
        if len(all_x_sub) > 0:
            data_dict[sub_id] = (np.concatenate(all_x_sub, axis=0), 
                                 np.concatenate(all_y_sub, axis=0))
            print(f"Loaded Subject {sub_id} | Total Nights: {len(all_x_sub)} | Total Epochs: {len(data_dict[sub_id][1])}")
            
    return data_dict

class SeqSleepDataset(Dataset):
    def __init__(self, X, y, seq_len=5):
        """
        X: (N, C, L) 所有的 EEG 信号片段
        y: (N,) 对应的标签
        seq_len: 序列长度（必须为奇数，如 5, 意味着前后各看 2 个 epoch）
        """
        self.X = X
        self.y = y
        self.seq_len = seq_len
        self.pad_len = seq_len // 2
        
        # 对首尾进行 Padding，保证第一个和最后一个 Epoch 也能作为中心点被预测
        # 信号特征 padding 0，标签 padding -1 (不过标签只取中心点，不会用到 pad 的值)
        self.X_padded = np.pad(self.X, ((self.pad_len, self.pad_len), (0, 0), (0, 0)), mode='constant')
        
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        # 取以 idx 为中心的序列
        start_idx = idx
        end_idx = idx + self.seq_len
        
        X_seq = self.X_padded[start_idx:end_idx] # (seq_len, C, L)
        y_center = self.y[idx]                   # 中心 Epoch 的标签
        
        return torch.FloatTensor(X_seq), torch.LongTensor([y_center])[0]

def create_dataloader(data_dict, sub_ids, batch_size=16, shuffle=True, num_workers=0, seq_len=5):
    """合并受试者并创建支持时序上下文的 DataLoader"""
    # 提取并拼接指定受试者的数据
    all_X = np.concatenate([data_dict[sid][0] for sid in sub_ids if sid in data_dict], axis=0)
    all_y = np.concatenate([data_dict[sid][1] for sid in sub_ids if sid in data_dict], axis=0)
    
    # 使用自定义的序列 Dataset
    dataset = SeqSleepDataset(all_X, all_y, seq_len=seq_len)
    
    loader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=shuffle, 
        num_workers=num_workers,
        drop_last=False
    )
    return loader

