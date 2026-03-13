import os
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from pyedflib import EdfReader
from scipy.signal import butter, filtfilt
import mne
from mne.decoding import Scaler

# --- 1. 单个受试者/记录读取与预处理 ---

def compute_plv_offline(x_raw):
    """
    预计算整晚数据的 PLV 矩阵。
    x_raw: numpy array (N, C, T)
    返回: numpy array (N, C, C)
    """
    # 避免负 stride 的 numpy 视图导致 torch 报错，先拷贝为连续数组
    if not isinstance(x_raw, np.ndarray):
        x_raw = np.array(x_raw)
    x_raw = np.ascontiguousarray(x_raw)
    x_tensor = torch.from_numpy(x_raw.astype(np.float32, copy=False))
    N, C, T = x_tensor.shape
    Xf = torch.fft.fft(x_tensor, dim=-1)
    h = torch.zeros(T, device=x_tensor.device) 
    
    if T % 2 == 0: 
        h[0] = h[T//2] = 1 
        h[1:T//2] = 2 
    else:
        h[0] = 1; h[1:(T+1)//2] = 2
    
    analytic_signal = torch.fft.ifft(Xf * h, dim=-1)
    phase = torch.angle(analytic_signal) 
    
    phase_diff = phase.unsqueeze(2) - phase.unsqueeze(1) 
    real_avg = torch.cos(phase_diff).mean(dim=-1)
    imag_avg = torch.sin(phase_diff).mean(dim=-1) 
    plv_matrix = torch.sqrt(real_avg**2 + imag_avg**2) 
    return plv_matrix.numpy()

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
        sleep_indices = np.where((labels > 0))[0]
        
        if len(sleep_indices) > 0:
            first_sleep = sleep_indices[0]
            last_sleep = sleep_indices[-1]
            buffer = 60 
            
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

def apply_subject_scaler_mne(X_sub, fs, channels):
    """
    使用 MNE 对受试者的所有数据（通常是两晚）进行通道级 z-score 归一化
    X_sub: numpy array (Total_Epochs, C, T)
    """
    # 1. 构建 MNE Info 对象
    # 根据你的 CHANNELS 确定类型，Sleep-EDF 通常是 eeg 或 eog
    ch_types = ['eeg'] * len(channels)
    info = mne.create_info(ch_names=channels, sfreq=fs, ch_types=ch_types)
    
    # 2. 使用 mne.decoding.Scaler
    # scalings='std' 对应 z-score ( (x - mean) / std )
    # 它会针对每个 channel 计算所有 epoch 的统计量
    scaler = Scaler(info=info, scalings='std')
    
    # Scaler 期望输入为 (n_epochs, n_channels, n_times)
    X_scaled = scaler.fit_transform(X_sub)
    
    return X_scaled.astype(np.float32)

# --- 2. 数据组织函数 ---

def get_subject_data_dict(root_path, subject_list, channels):
    data_dict = {}
    all_files = os.listdir(root_path)
    
    for sub_id in subject_list:
        psg_files = sorted([f for f in all_files if f.startswith(sub_id) and f.endswith("PSG.edf")])
        
        all_x_raw_sub = [] # 存储未归一化的原始信号
        all_y_sub = []
        
        # 第一步：先加载该受试者所有的夜晚数据
        for psg_f in psg_files:
            file_prefix = psg_f[:6]
            hyp_f = [f for f in all_files if f.startswith(file_prefix) and f.endswith("Hypnogram.edf")]
            
            if len(hyp_f) > 0:
                psg_path = os.path.join(root_path, psg_f)
                hyp_path = os.path.join(root_path, hyp_f[0])
                
                # 注意：修改 load_edf_record，内部不再调用 apply_scaler
                X, y, fs = load_edf_record(psg_path, hyp_path, channels=channels)
                
                if X is not None:
                    # 只做基础带通滤波，不做归一化
                    X = sleep_bandpass(X, fs)
                    all_x_raw_sub.append(X)
                    all_y_sub.append(y)

        # 第二步：如果该受试者有数据，合并后进行受试者级归一化
        if len(all_x_raw_sub) > 0:
            X_combined = np.concatenate(all_x_raw_sub, axis=0) # (Total_Epochs, C, T)
            Y_combined = np.concatenate(all_y_sub, axis=0)
            
            # 使用 MNE 进行受试者级归一化
            X_scaled = apply_subject_scaler_mne(X_combined, fs, channels)
            
            # 第三步：归一化后再计算 PLV（保证相位特征的提取基于标准化信号，更稳定）
            A_plv = compute_plv_offline(X_scaled)
            
            data_dict[sub_id] = (X_scaled, Y_combined, A_plv.astype(np.float32))
            print(f"Loaded Subject {sub_id} | Subject-wise Scaled | Total Epochs: {len(Y_combined)}")
            
    return data_dict

class SeqSleepDataset(Dataset):
    def __init__(self, X, y, A_plv, seq_len=5):
        self.X = X
        self.y = y
        self.A_plv = A_plv
        self.seq_len = seq_len
        self.pad_len = seq_len // 2
        
        # 信号和 PLV 矩阵同步 Padding
        self.X_padded = np.pad(self.X, ((self.pad_len, self.pad_len), (0, 0), (0, 0)), mode='constant')
        self.A_plv_padded = np.pad(self.A_plv, ((self.pad_len, self.pad_len), (0, 0), (0, 0)), mode='constant')
        
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return (torch.FloatTensor(self.X_padded[idx:idx+self.seq_len]), 
                torch.LongTensor([self.y[idx]])[0], 
                torch.FloatTensor(self.A_plv_padded[idx:idx+self.seq_len]))
def create_dataloader(data_dict, sub_ids, batch_size=16, shuffle=True, num_workers=0, seq_len=5):
    """合并受试者并创建支持时序上下文的 DataLoader"""
    all_X = np.concatenate([data_dict[sid][0] for sid in sub_ids if sid in data_dict], axis=0)
    all_y = np.concatenate([data_dict[sid][1] for sid in sub_ids if sid in data_dict], axis=0)
    all_aplv = np.concatenate([data_dict[sid][2] for sid in sub_ids if sid in data_dict], axis=0)
    
    dataset = SeqSleepDataset(all_X, all_y, all_aplv, seq_len=seq_len)
    
    loader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=shuffle, 
        num_workers=num_workers,
        drop_last=False
    )
    return loader

