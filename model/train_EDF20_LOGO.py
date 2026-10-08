import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import confusion_matrix, classification_report, cohen_kappa_score, f1_score, accuracy_score

import matplotlib
matplotlib.use('Agg') # 保证在无显示器服务器上运行
import matplotlib.pyplot as plt
import seaborn as sns

from model import SleepGATNet
from SleepEDFdataset import get_data_dict, SeqSleepDataset

# ==========================================
# 1. 核心配置
# ==========================================
CONFIG = {
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "dataset_root": r"E:\EEG\dataset\sleep-edf-database-expanded-1.0.0\SleepEDF-20",
    "channels": ["EEG Fpz-Cz", "EEG Pz-Oz", "EOG horizontal"],
    "fs": 100,
    "batch_size": 32, 
    "epochs": 60,
    "lr": 2e-4,           # 较低起步，配合余弦退火
    "weight_decay": 1e-3, 
    "patience": 8,       # 早停耐心值
    "seq_len": 21,        # 捕捉长程上下文
    "save_dir": "./checkpoints_loso_edf20",
    "log_dir": "./runs/sleep_gat_final"
}

# ==========================================
# 2. 增强型评估与绘图函数
# ==========================================
def plot_final_confusion_matrix(y_true, y_pred, save_path):
    class_names = ['W', 'N1', 'N2', 'N3', 'REM']
    cm = confusion_matrix(y_true, y_pred, labels=range(5))
    # 归一化显示百分比
    cm_percent = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-9) * 100
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm_percent, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Global Confusion Matrix (20 Subjects Overall) [%]')
    plt.xlabel('Predicted Stage')
    plt.ylabel('True Stage')
    plt.savefig(save_path, dpi=300)
    plt.close() # 彻底释放内存

class EarlyStopping:
    def __init__(self, patience=10, path='checkpoint.pt'):
        self.patience, self.counter, self.best_score, self.early_stop, self.path = patience, 0, None, False, path
    def __call__(self, val_acc, model):
        if self.best_score is None or val_acc > self.best_score:
            self.best_score, self.counter = val_acc, 0
            torch.save(model.state_dict(), self.path)
        else:
            self.counter += 1
            if self.counter >= self.patience: self.early_stop = True

# ==========================================
# 3. LOSO 训练主程序
# ==========================================
def train_loso():
    os.makedirs(CONFIG["save_dir"], exist_ok=True)
    
    # A. 数据加载
    print(">>> 正在初始化受试者数据...")
    sub_list = sorted({f[:5] for f in os.listdir(CONFIG["dataset_root"]) if "PSG.edf" in f})[:20]
    data_dict, P_matrix, _ = get_data_dict(CONFIG["dataset_root"], sub_list, CONFIG["channels"])
    P_matrix = P_matrix.to(CONFIG["device"])
    
    # 损失函数：给 N1 分配 4 倍权重，W/N2 适当降低
    class_weights = torch.tensor([1.0, 4.0, 0.7, 1.2, 1.2]).to(CONFIG["device"])
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # 用于存放所有人的总结果
    global_y_true, global_y_pred = [], []
    # 用于存放每个受试者的指标
    subject_results = []

    # B. 20折 LOSO
    for fold, test_sub in enumerate(sub_list):
        print(f"\n{'#'*15} FOLD {fold+1}/20: Testing Subject {test_sub} {'#'*15}")
        
        train_subs = [s for s in sub_list if s != test_sub]
        train_set = SeqSleepDataset(data_dict, train_subs, seq_len=CONFIG["seq_len"])
        test_set = SeqSleepDataset(data_dict, [test_sub], seq_len=CONFIG["seq_len"])
        
        train_loader = DataLoader(train_set, batch_size=CONFIG["batch_size"], shuffle=True)
        test_loader = DataLoader(test_set, batch_size=CONFIG["batch_size"])

        # 初始化模型
        model = SleepGATNet(channel_names=CONFIG["channels"], fs=CONFIG["fs"]).to(CONFIG["device"])
        optimizer = optim.AdamW(model.parameters(), lr=CONFIG["lr"], weight_decay=CONFIG["weight_decay"])
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG["epochs"])
        writer = SummaryWriter(os.path.join(CONFIG["log_dir"], f"fold_{test_sub}"))
        
        save_path = os.path.join(CONFIG["save_dir"], f"best_{test_sub}.pt")
        early_stopping = EarlyStopping(patience=CONFIG["patience"], path=save_path)

        for epoch in range(CONFIG["epochs"]):
            model.train()
            total_loss, grad_norms = 0.0, []
            
            for x, y, a_fc in train_loader:
                x, y, a_fc = x.to(CONFIG["device"]), y.to(CONFIG["device"]), a_fc.to(CONFIG["device"])
                optimizer.zero_grad()
                
                # forward: 得到主预测 logits 和 TRGAT 辅助 logits
                logits, _, _ = model(x, a_fc, P_matrix)
                
                # 计算损失 (主损失 + 辅助损失)
                loss = criterion(logits[:, CONFIG["seq_len"]//2, :], y)

                
                loss.backward()
                
                # 梯度裁剪与监控
                gnorm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                grad_norms.append(gnorm.item())
                
                optimizer.step()
                total_loss += loss.item()
            
            # 记录 TensorBoard
            writer.add_scalar('Loss/Train', total_loss/len(train_loader), epoch)
            writer.add_scalar('Gradient/MaxNorm', np.max(grad_norms), epoch)

            # 在当前测试受试者上跑一遍验证
            model.eval()
            correct, v_total = 0, 0
            with torch.no_grad():
                for vx, vy, va_fc in test_loader:
                    vx, vy, va_fc = vx.to(CONFIG["device"]), vy.to(CONFIG["device"]), va_fc.to(CONFIG["device"])
                    out, _, _ = model(vx, va_fc, P_matrix)
                    pred = out[:, CONFIG["seq_len"]//2, :].argmax(dim=1)
                    correct += (pred == vy).sum().item()
                    v_total += vy.size(0)
            
            val_acc = correct / v_total
            writer.add_scalar('Acc/TestSub', val_acc, epoch)
            scheduler.step()
            
            if (epoch+1) % 5 == 0:
                print(f"  Epoch {epoch+1:02d} | Loss: {total_loss/len(train_loader):.4f} | Acc: {val_acc:.4f}")
            
            early_stopping(val_acc, model)
            if early_stopping.early_stop: break

        # C. 统计单人战果
        model.load_state_dict(torch.load(save_path))
        model.eval()
        f_true, f_pred = [], []
        with torch.no_grad():
            for tx, ty, ta_fc in test_loader:
                tx, ty, ta_fc = tx.to(CONFIG["device"]), ty.to(CONFIG["device"]), ta_fc.to(CONFIG["device"])
                out, _, _ = model(tx, ta_fc, P_matrix)
                pred = out[:, CONFIG["seq_len"]//2, :].argmax(dim=1)
                f_true.extend(ty.cpu().numpy()); f_pred.extend(pred.cpu().numpy())
        
        # 计算该受试者的核心指标
        f_acc = accuracy_score(f_true, f_pred)
        f_f1 = f1_score(f_true, f_pred, average='macro')
        f_kappa = cohen_kappa_score(f_true, f_pred)
        
        subject_results.append({
            "Subject": test_sub,
            "Accuracy": f_acc,
            "Macro-F1": f_f1,
            "Kappa": f_kappa
        })
        
        global_y_true.extend(f_true)
        global_y_pred.extend(f_pred)
        writer.close()

    # ==========================================
    # 4. 打印“成绩单”表格与最终热力图
    # ==========================================
    df = pd.DataFrame(subject_results)
    print("\n" + "="*60)
    print("SUBJECT-WISE PERFORMANCE TABLE")
    print("="*60)
    print(df.to_string(index=False))
    df.to_csv(os.path.join(CONFIG["save_dir"], "subject_metrics.csv"), index=False)
    
    print("\n" + "="*60)
    print("GLOBAL CLASSIFICATION REPORT")
    print("="*60)
    print(classification_report(global_y_true, global_y_pred, target_names=['W', 'N1', 'N2', 'N3', 'REM']))
    
    # 最终绘图
    plot_final_confusion_matrix(global_y_true, global_y_pred, os.path.join(CONFIG["save_dir"], "final_global_cm.png"))

if __name__ == "__main__":
    train_loso()


