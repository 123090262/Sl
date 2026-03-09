import os
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
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

def create_dataloader(data_dict, sub_ids, batch_size=16, shuffle=True,num_workers = 8):
    """合并受试者并创建 DataLoader (同 ISRUC)"""
    all_X = np.concatenate([data_dict[sid][0] for sid in sub_ids if sid in data_dict], axis=0)
    all_y = np.concatenate([data_dict[sid][1] for sid in sub_ids if sid in data_dict], axis=0)
    
    tensor_x = torch.tensor(all_X, dtype=torch.float32)
    tensor_y = torch.tensor(all_y).long()
    
    dataset = TensorDataset(tensor_x, tensor_y)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

# --- 3. 快速测试 (可选) ---
def print_dataset_statistics(data_dict):
    """
    统计并输出数据库各阶段样本分布表格
    """
    all_y = np.concatenate([v[1] for v in data_dict.values()], axis=0)
    unique, counts = np.unique(all_y, return_counts=True)
    stats = dict(zip(unique, counts))
    
    stage_names = {0: "W (Wake)", 1: "N1", 2: "N2", 3: "N3 (Slow Wave)", 4: "REM"}
    
    print("\n" + "="*45)
    print(f"{'Sleep Stage':<20} | {'Count':<10} | {'Percentage':<10}")
    print("-" * 45)
    
    total = len(all_y)
    for i in range(5):
        count = stats.get(i, 0)
        percentage = (count / total) * 100
        name = stage_names.get(i, "Unknown")
        print(f"{name:<20} | {count:<10} | {percentage:>8.2f}%")
    
    print("-" * 45)
    print(f"{'Total Epochs':<20} | {total:<10} | 100.00%")
    print("="*45 + "\n")

# --- 主程序逻辑修改 ---
if __name__ == "__main__":
    ROOT = r"E:\EEG\dataset\sleep-edf-database-expanded-1.0.0\SleepEDF-78"
    SUBS = [f"SC4{i:02d}" for i in range(78)] # 可根据实际文件夹填充
    CHNS = ["EEG Fpz-Cz", "EEG Pz-Oz"]
    
    data = get_subject_data_dict(ROOT, SUBS, CHNS)
    
    if data:
        # 输出统计表格
        print_dataset_statistics(data)
        
        loader = create_dataloader(data, SUBS)
        # 后续训练代码...