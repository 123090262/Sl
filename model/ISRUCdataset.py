import os
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from pyedflib import EdfReader
from scipy.signal import butter, filtfilt
from sklearn.model_selection import train_test_split, LeaveOneOut

# --- 1. 单个受试者读取与预处理 ---

def load_isruc_rec(subject_dir, expert=1, channels=None):
    rec_files = [f for f in os.listdir(subject_dir) if f.endswith(".rec")]
    if len(rec_files) == 0: 
        raise FileNotFoundError("未找到 .rec 文件")
    
    rec_path = os.path.join(subject_dir, rec_files[0])
    label_path = rec_path.replace(".rec", f"_{expert}.txt")
    
    if not os.path.exists(label_path): 
        raise FileNotFoundError(f"未找到标签文件：{label_path}")

    edf = EdfReader(rec_path)
    fs = int(edf.getSampleFrequency(0))
    all_labels = edf.getSignalLabels()
    n_channels = edf.signals_in_file
    
    if channels is not None:
        idx = [all_labels.index(ch) for ch in channels] #根据指定通道名呈建立索引
    else:
        idx = list(range(n_channels)) #若无指定，提取全部
    
    signals = []
    for i in idx:
        sig = edf.readSignal(i) #一维数组，（T，）
        signals.append(sig)
    edf._close()

    data = np.vstack(signals) #（C,T）
    samples_per_epoch = 30 * fs
    num_epochs = data.shape[1] // samples_per_epoch #T/
    data = data[:, :num_epochs * samples_per_epoch]
    X = data.reshape(len(idx), num_epochs, samples_per_epoch).transpose(1, 0, 2) #（B,C,T）

    label_map = {'W':0, '0':0, 
                 'N1':1, '1':1, 
                 'N2':2, '2':2, 
                 'N3':3, '3':3, 
                 'REM':4, 'R':4, '5':4}
    
    with open(label_path, "r") as f:
        raw_y = [l.strip() for l in f.readlines()]
    
    y = [label_map[raw_y[i]] if i < len(raw_y) and raw_y[i] in label_map else 0 for i in range(num_epochs)]
    return X.astype(np.float32), np.array(y), fs

def sleep_bandpass(data, fs, lowcut=0.3, highcut=35.0, order=4):
    nyq = 0.5 * fs
    b, a = butter(order, [lowcut/nyq, highcut/nyq], btype='band')
    return filtfilt(b, a, data, axis=-1)

def apply_scaler(X):
    # 按通道进行标准化
    for c in range(X.shape[1]):
        mean, std = X[:, c, :].mean(), X[:, c, :].std()
        X[:, c, :] = (X[:, c, :] - mean) / (std + 1e-8)
    return X

# --- 2. 数据组织函数 ---
def get_subject_data_dict(root_path, subject_list, channels):
    """将每个受试者的数据分开存储在字典中"""
    data_dict = {}
    
    for sub_id in subject_list: #1-100 for S1, 1-10 for S3
        sub_path = os.path.join(root_path, sub_id)
        X, y, fs = load_isruc_rec(sub_path, channels=channels)
        if X is not None:
            X = sleep_bandpass(X, fs)
            X = apply_scaler(X)
            X = X.astype(np.float32)
            data_dict[sub_id] = (X, y)
            print(f"Loaded {sub_id}")
    return data_dict 

def create_dataloader(data_dict, sub_ids, batch_size=32, shuffle=True):
    """根据给定的受试者ID列表合并数据并创建 DataLoader"""
    all_X = np.concatenate([data_dict[sid][0] for sid in sub_ids], axis=0) #对当前训练集的所有受试者拼接
    all_y = np.concatenate([data_dict[sid][1] for sid in sub_ids], axis=0)
    
    tensor_x = torch.tensor(all_X,dtype=torch.float32)
    tensor_y = torch.tensor(np.array(all_y), dtype=torch.long)
    
    dataset = TensorDataset(tensor_x, tensor_y)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

