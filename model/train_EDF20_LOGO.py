import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import copy
import os
import numpy as np
import json
from sklearn.metrics import confusion_matrix, classification_report, cohen_kappa_score, roc_auc_score
from model import SleepGATNet 
from SleepEDFdataset import get_subject_data_dict, create_dataloader
import seaborn as sns

# --- 1. 配置参数 ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATASET_ROOT = "E:\EEG\dataset\sleep-edf-database-expanded-1.0.0\SleepEDF-20"
CHANNELS = ["EEG Fpz-Cz", "EEG Pz-Oz"] 
FS = 100  
BATCH_SIZE = 64
EPOCHS = 80
LR = 1e-4
SAVE_DIR = "./checkpoints_edf_loso" # 修改目录名以区分
os.makedirs(SAVE_DIR, exist_ok=True)

# --- 2. 绘图与指标辅助函数 ---
def plot_confusion_matrix(y_true, y_pred, title='Confusion Matrix', save_path=None):
    class_names = ['W', 'N1', 'N2', 'N3', 'REM']
    cm = confusion_matrix(y_true, y_pred, labels=range(len(class_names)))
    cm_percent = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-9) * 100
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_percent, annot=True, fmt='.2f', cmap='Greens',
                xticklabels=class_names, yticklabels=class_names)
    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.close()

def run_loso_fold(fold_idx, train_loader, test_loader):
    # 针对 Sleep-EDF 和 ISRUC 等生理信号任务
    model = SleepGATNet(channel_names=CHANNELS, fs=FS).to(DEVICE)

    # --- 动态根据训练集真实类别频率计算 Class Weights ---
    num_classes = 5  # ['W', 'N1', 'N2', 'N3', 'REM']
    class_counts = np.zeros(num_classes, dtype=np.float32)
    # 直接遍历一次 train_loader 统计标签分布，兼容任意 Dataset 类型
    for _, y_batch in train_loader:
        y_np = y_batch.cpu().numpy().astype(np.int64)
        class_counts += np.bincount(y_np, minlength=num_classes).astype(np.float32)

    class_freq = class_counts / (class_counts.sum() + 1e-9)
    inv_freq = 1.0 / (class_freq + 1e-9)
    class_weights = inv_freq / inv_freq.mean()  # 归一化到均值约为 1
    class_weights = torch.tensor(class_weights, dtype=torch.float32, device=DEVICE)

    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # --- 引入学习率调度器（基于验证集 Loss 自适应衰减）---
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=3,
    )

    history = {'train_loss':[], 'train_acc': [], 'val_loss': [], 'val_acc':[]}
    best_val_loss = float('inf')
    best_state = None
    patience = 10
    no_improve = 0

    for epoch in range(EPOCHS):
        model.train()
        train_loss, correct_train, total_train = 0.0, 0, 0
        
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)

            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            correct_train += (preds == y).sum().item()
            total_train += y.size(0)
            
        # 第一次 eval：用于监控训练过程中的验证集表现
        model.eval()
        val_loss, correct_val, total_val = 0.0, 0, 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                logits = model(x)
                loss_val = criterion(logits, y)
                val_loss += loss_val.item()
                preds = torch.argmax(logits, dim=1)
                correct_val += (preds == y).sum().item()
                total_val += y.size(0)

        history['train_loss'].append(train_loss / len(train_loader))
        history['train_acc'].append(correct_train / total_train)
        avg_val_loss = val_loss / len(test_loader)
        history['val_loss'].append(avg_val_loss)
        history['val_acc'].append(correct_val / total_val)

        # 根据验证集 Loss 调整学习率
        scheduler.step(avg_val_loss)

        # 早停：基于验证集 loss
        if avg_val_loss < best_val_loss - 1e-4:
            best_val_loss = avg_val_loss
            best_state = copy.deepcopy(model.state_dict())
            no_improve = 0
        else:
            no_improve += 1

        if (epoch + 1) % 5 == 0:
            print(f"Fold {fold_idx} | Ep {epoch+1}/{EPOCHS} | Val Acc: {history['val_acc'][-1]:.4f} | Val Loss: {avg_val_loss:.4f}")

        if no_improve >= patience:
            print(f"Fold {fold_idx} | Early stopping at epoch {epoch+1}, best Val Loss: {best_val_loss:.4f}")
            break

    # 加载最佳验证表现的权重
    if best_state is not None:
        model.load_state_dict(best_state)

    # 第二次 eval：训练完成后，提取最终预测结果用于统计
    model.eval()
    y_true_fold, y_pred_fold, y_prob_fold = [], [], []
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            logits = model(x)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)
            
            y_true_fold.extend(y.cpu().numpy())
            y_pred_fold.extend(preds.cpu().numpy())
            y_prob_fold.extend(probs.cpu().numpy())
            
    return np.array(y_true_fold), np.array(y_pred_fold), np.array(y_prob_fold), history, model

if __name__ == "__main__":
    # 20个受试者：SC400 - SC419
    subjects = [f"SC4{i:02d}" for i in range(20)] 
    all_data = get_subject_data_dict(DATASET_ROOT, subjects, CHANNELS)

    all_y_true, all_y_pred, all_y_prob = [], [], []
    all_history = [] 

    # --- 修改为 20 折循环 (Leave-One-Subject-Out) ---
    for i, test_sub in enumerate(subjects):
        train_subs = [s for s in subjects if s != test_sub]
        
        print(f"\n>>>> Fold {i+1}/20 | Testing on Subject: {test_sub}")
        train_loader = create_dataloader(all_data, train_subs, batch_size=BATCH_SIZE)
        test_loader = create_dataloader(all_data, [test_sub], batch_size=BATCH_SIZE, shuffle=False)
        
        y_true, y_pred, y_prob, history, trained_model = run_loso_fold(i+1, train_loader, test_loader)
        
        # 保存本地权重，方便后续做 CAM 可视化
        model_save_path = os.path.join(SAVE_DIR, f"edf20_loso_model_{test_sub}.pth")
        torch.save(trained_model.state_dict(), model_save_path)
        
        all_y_true.extend(y_true)
        all_y_pred.extend(y_pred)
        all_y_prob.extend(y_prob)
        all_history.append(history)

    # --- 最终指标计算 ---
    print("\n" + "="*60)
    print("FINAL LOSO RESULTS (Sleep-EDF 20 Subjects)")
    
    # 1. 基础分类报告
    print(classification_report(all_y_true, all_y_pred, target_names=['W', 'N1', 'N2', 'N3', 'REM'], digits=4))
    
    # 2. Cohen's Kappa 
    kappa = cohen_kappa_score(all_y_true, all_y_pred)
    print(f"Overall Cohen's Kappa: {kappa:.4f}")
    
    # 3. Macro AUROC
    macro_auroc = roc_auc_score(all_y_true, all_y_prob, multi_class='ovo')
    print(f"Overall Macro AUROC: {macro_auroc:.4f}")
    
    # 4. 混淆矩阵
    plot_confusion_matrix(all_y_true, all_y_pred, title='Overall Sleep-EDF LOSO CM', save_path=f'{SAVE_DIR}/edf_loso_cm.png')
    
    # 保存预测结果
    np.savez(f"{SAVE_DIR}/edf20_loso_preds.npz", y_true=all_y_true, y_pred=all_y_pred, y_prob=all_y_prob)
    print(f"\n[任务完成] 结果已保存至 {SAVE_DIR}")