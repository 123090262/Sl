import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import os
import numpy as np
import json
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix, classification_report, cohen_kappa_score, roc_auc_score
from model import SleepGATNet
from SleepEDFdataset import get_subject_data_dict, create_dataloader
import seaborn as sns

# ================= 1. 配置参数 =================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATASET_ROOT = "D:/sleep_project/dataset/SleepEDF-78" # 请根据本地实际路径修改
CHANNELS = ["EEG Fpz-Cz", "EEG Pz-Oz"]
FS = 100
BATCH_SIZE = 64             
EPOCHS = 30
LR = 1e-3

SAVE_DIR = "./checkpoints_edf78_loso"
os.makedirs(SAVE_DIR, exist_ok=True)

# 缓存目录：用于存放 .pt 文件，避免重复解析 EDF
CACHE_DIR = "D:/sleep_project/dataset/SleepEDF_fast_cache"
os.makedirs(CACHE_DIR, exist_ok=True)

# ================= 2. 辅助函数 =================
def print_dataset_statistics(data_dict):
    """
    统计并输出全数据库各阶段样本分布表格
    """
    all_y = np.concatenate([v[1] for v in data_dict.values()], axis=0)
    unique, counts = np.unique(all_y, return_counts=True)
    stats = dict(zip(unique, counts))
    
    stage_names = {0: "W (Wake)", 1: "N1", 2: "N2", 3: "N3 (Slow Wave)", 4: "REM"}
    
    print("\n" + "="*50)
    print(f"{'Sleep Stage':<20} | {'Count':<10} | {'Percentage':<10}")
    print("-" * 50)
    
    total = len(all_y)
    for i in range(5):
        count = stats.get(i, 0)
        percentage = (count / total) * 100
        name = stage_names.get(i, "Unknown")
        print(f"{name:<20} | {count:<10} | {percentage:>8.2f}%")
    
    print("-" * 50)
    print(f"{'Total Epochs':<20} | {total:<10} | 100.00%")
    print("="*50 + "\n")

def plot_confusion_matrix(y_true, y_pred, title='Confusion Matrix', save_path=None):
    class_names = ['W', 'N1', 'N2', 'N3', 'REM']
    cm = confusion_matrix(y_true, y_pred, labels=range(len(class_names)))
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_percent, annot=True, fmt='.2f', cmap='Greens', xticklabels=class_names, yticklabels=class_names)
    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.close()

# --- 3. LOGO 训练逻辑 ---

def fast_load_all_data(dataset_root, subs, channels):
    """
    【预加载核心】一次性载入所有被试数据到内存
    """
    all_data = {}
    subs_to_parse = []
    
    # 检查缓存
    for sub in subs:
        cache_path = os.path.join(CACHE_DIR, f"{sub}.pt")
        if os.path.exists(cache_path):
            all_data[sub] = torch.load(cache_path)
        else:
            subs_to_parse.append(sub)
            
    if subs_to_parse:
        print(f"[*] 缓存未命中，正在解析 {len(subs_to_parse)} 个被试的原始 EDF 文件...")
        slow_data = get_subject_data_dict(dataset_root, subs_to_parse, channels)
        for sub, data in slow_data.items():
            all_data[sub] = data
            torch.save(data, os.path.join(CACHE_DIR, f"{sub}.pt"))
            
    print(f"[*] 成功载入 {len(all_data)} 个被试的数据。")
    return all_data

# ================= 3. 训练逻辑 (与 EDF20 一致) =================
def run_logo_fold(fold_idx, train_loader, test_loader):
    model = SleepGATNet(channel_names=CHANNELS, fs=FS).to(DEVICE)
    class_weights = torch.tensor([1.0, 2.0, 1.0, 1.0, 1.0]).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    
    history = {'train_loss':[], 'train_acc':[], 'val_loss':[], 'val_acc':[]}
    
    for epoch in range(EPOCHS):
        model.train()
        train_loss, correct_train, total_train = 0.0, 0, 0
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            logits, _, _, _ = model(x)
            loss = criterion(logits, y) 
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            correct_train += (preds == y).sum().item()
            total_train += y.size(0)
            
        model.eval()
        val_loss, correct_val, total_val = 0.0, 0, 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                logits, _, _, _ = model(x)
                val_loss += criterion(logits, y).item()
                preds = torch.argmax(logits, dim=1)
                correct_val += (preds == y).sum().item()
                total_val += y.size(0)

        history['train_loss'].append(train_loss / len(train_loader))
        history['train_acc'].append(correct_train / total_train)
        history['val_loss'].append(val_loss / len(test_loader))
        history['val_acc'].append(correct_val / total_val)
        
        if (epoch + 1) % 5 == 0:
            print(f"Fold {fold_idx} | Epoch {epoch+1:02d} | Train Acc: {history['train_acc'][-1]:.4f} | Val Acc: {history['val_acc'][-1]:.4f}")

    # 获取预测结果
    y_true_fold, y_pred_fold, y_prob_fold = [], [], []
    model.eval()
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            logits, _, _, _ = model(x)
            y_true_fold.extend(y.cpu().numpy())
            y_pred_fold.extend(torch.argmax(logits, dim=1).cpu().numpy())
            y_prob_fold.extend(torch.softmax(logits, dim=1).cpu().numpy())
            
    return np.array(y_true_fold), np.array(y_pred_fold), np.array(y_prob_fold), history, model

# ================= 4. 主程序入口 =================
if __name__ == "__main__":
    # 78 名被试的基础编号（按 person 分折，确保同一个人的两晚永远在同一侧）
    unique_persons = sorted([f"SC4{i:02d}" for i in range(78)])
    all_subs = [p + "1" for p in unique_persons] + [p + "2" for p in unique_persons]

    # A. 一次性加载全库数据
    all_data = fast_load_all_data(DATASET_ROOT, all_subs, CHANNELS)

    # B. 输出全库统计表格
    print_dataset_statistics(all_data)

    # C. 20 折交叉验证（按 person 划分，再映射到 night1/night2）
    n_splits = 20
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    all_y_true, all_y_pred, all_y_prob = [], [], []
    all_history = []

    for fold_idx, (train_person_idx, test_person_idx) in enumerate(kf.split(unique_persons)):
        print(f"\n>>>> Fold {fold_idx + 1}/{n_splits}")

        train_persons = [unique_persons[i] for i in train_person_idx]
        test_persons = [unique_persons[i] for i in test_person_idx]

        train_subs = [p + "1" for p in train_persons] + [p + "2" for p in train_persons]
        test_subs = [p + "1" for p in test_persons] + [p + "2" for p in test_persons]

        train_loader = create_dataloader(all_data, train_subs, batch_size=BATCH_SIZE)
        test_loader = create_dataloader(all_data, test_subs, batch_size=BATCH_SIZE, shuffle=False)

        y_true, y_pred, y_prob, history, model = run_logo_fold(fold_idx + 1, train_loader, test_loader)

        torch.save(model.state_dict(), os.path.join(SAVE_DIR, f"edf78_fold_{fold_idx+1}.pth"))

        all_y_true.extend(y_true)
        all_y_pred.extend(y_pred)
        all_y_prob.extend(y_prob)
        all_history.append(history)

    # D. 汇总结果
    print("\n" + "="*60)
    print(f"FINAL {n_splits}-FOLD CV RESULTS (Sleep-EDF-78)")
    print(classification_report(all_y_true, all_y_pred, target_names=['W', 'N1', 'N2', 'N3', 'REM'], digits=4))

    kappa = cohen_kappa_score(all_y_true, all_y_pred)
    print(f"Overall Cohen's Kappa: {kappa:.4f}")

    try:
        macro_auroc = roc_auc_score(all_y_true, all_y_prob, multi_class='ovo')
        print(f"Overall Macro AUROC: {macro_auroc:.4f}")
    except Exception as e:
        print(f"Overall Macro AUROC: N/A ({e})")

    plot_confusion_matrix(all_y_true, all_y_pred, title='Overall Sleep-EDF-78 CV CM', save_path=f'{SAVE_DIR}/edf78_final_cm.png')
    np.savez(f"{SAVE_DIR}/edf78_results.npz", y_true=all_y_true, y_pred=all_y_pred, y_prob=all_y_prob)