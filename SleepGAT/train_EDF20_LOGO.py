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
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import seaborn as sns
import torch.nn.functional as F # 👉👉👉 【修改：新增导入】 👈👈👈

from model import SleepGATNet
from SleepEDFdataset import get_data_dict, get_transition_matrix, SeqSleepDataset

# ==========================================
# 1. 核心配置
# ==========================================
CONFIG = {
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "dataset_root": r"E:\EEG\dataset\sleep-edf-database-expanded-1.0.0\SleepEDF-20",
    "channels": ["EEG Fpz-Cz", "EEG Pz-Oz", "EOG horizontal"],
    "fs": 100,
    "batch_size": 64, 
    "epochs": 60,
    "lr": 2e-4,
    "weight_decay": 5e-4, 
    "patience": 20,
    "seq_len": 7,
    "save_dir": "./checkpoints_debug_412_417", # 👉👉👉 【修改：更改保存目录】 👈👈👈
    "log_dir": "./runs/debug_focal_loss"
}

# 👉👉👉 【修改：新增 Focal Loss 定义】 👈👈👈
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, weight=self.alpha, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        if self.reduction == 'mean': return focal_loss.mean()
        elif self.reduction == 'sum': return focal_loss.sum()
        else: return focal_loss

# 👉👉👉 【修改：新增 Loss 曲线绘制函数】 👈👈👈
def plot_loss_curve(train_losses, save_path, subject_id):
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(train_losses) + 1), train_losses, label='Training Loss', color='red', linewidth=2)
    plt.title(f'Training Loss Curve - Subject {subject_id}')
    plt.xlabel('Epochs')
    plt.ylabel('Loss Value')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_final_confusion_matrix(y_true, y_pred, save_path):
    class_names = ['W', 'N1', 'N2', 'N3', 'REM']
    cm = confusion_matrix(y_true, y_pred, labels=range(5))
    cm_percent = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-9) * 100
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm_percent, annot=True, fmt='.2f', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix [%]')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig(save_path, dpi=300)
    plt.close()

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
# 3. 调试版训练主程序
# ==========================================
def train_loso():
    os.makedirs(CONFIG["save_dir"], exist_ok=True)
    
    print(">>> 正在加载数据...")
    # 获取全部受试者列表
    all_subs = sorted({f[:5] for f in os.listdir(CONFIG["dataset_root"]) if "PSG.edf" in f})[:20]
    data_dict, _, _ = get_data_dict(CONFIG["dataset_root"], all_subs, CONFIG["channels"])
    
    # 👉👉👉 【修改：仅锁定 SC412 和 SC417 进行快速调试】 👈👈👈
    target_subs = [s for s in all_subs if "412" in s or "417" in s]
    print(f">>> 调试模式：将仅对以下受试者进行测试: {target_subs}")

    # 👉👉👉 【修改：配置 Focal Loss 权重】 👈👈👈
    # 类别顺序: W, N1, N2, N3, REM。增加 N1(1) 权重至 4.0，降低 W(0) 和 N2(2)
    alpha_weights = torch.tensor([0.6, 4.0, 0.6, 1.0, 1.0]).to(CONFIG["device"])
    criterion = FocalLoss(alpha=alpha_weights, gamma=2.0)

    global_y_true, global_y_pred = [], []
    subject_results = []

    for fold, test_sub in enumerate(target_subs):
        print(f"\n{'#'*15} DEBUG FOLD: Subject {test_sub} {'#'*15}")
        
        train_subs = [s for s in all_subs if s != test_sub]
        train_set = SeqSleepDataset(data_dict, train_subs, seq_len=CONFIG["seq_len"])
        test_set = SeqSleepDataset(data_dict, [test_sub], seq_len=CONFIG["seq_len"])
        P_matrix = get_transition_matrix(data_dict, train_subs).to(CONFIG["device"])
        
        train_loader = DataLoader(train_set, batch_size=CONFIG["batch_size"], shuffle=True)
        test_loader = DataLoader(test_set, batch_size=CONFIG["batch_size"])

        model = SleepGATNet(channel_names=CONFIG["channels"], fs=CONFIG["fs"]).to(CONFIG["device"])
        optimizer = optim.AdamW(model.parameters(), lr=CONFIG["lr"], weight_decay=CONFIG["weight_decay"])
        
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG["epochs"])
        writer = SummaryWriter(os.path.join(CONFIG["log_dir"], f"debug_{test_sub}"))
        
        save_path = os.path.join(CONFIG["save_dir"], f"best_{test_sub}.pt")
        early_stopping = EarlyStopping(patience=CONFIG["patience"], path=save_path)

        # 👉👉👉 【修改：新增列表用于存储每轮 Loss】 👈👈👈
        epoch_loss_history = []

        for epoch in range(CONFIG["epochs"]):
            model.train()
            total_loss = 0.0
                
            for x, y, a_fc in train_loader:
                x, y, a_fc = x.to(CONFIG["device"]), y.to(CONFIG["device"]), a_fc.to(CONFIG["device"])
                optimizer.zero_grad()
                logits, _ = model(x, a_fc, P_matrix)
                
                center_index = CONFIG["seq_len"] // 2
                loss = criterion(logits[:, center_index, :], y[:, center_index])
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                total_loss += loss.item()
            
            avg_epoch_loss = total_loss / len(train_loader)
            epoch_loss_history.append(avg_epoch_loss) # 👉👉👉 【记录 Loss】 👈👈👈
            
            writer.add_scalar('Loss/Train', avg_epoch_loss, epoch)

            # 验证
            model.eval()
            correct, v_total = 0, 0
            with torch.no_grad():
                for vx, vy, va_fc in test_loader:
                    vx, vy, va_fc = vx.to(CONFIG["device"]), vy.to(CONFIG["device"]), va_fc.to(CONFIG["device"])
                    out, _ = model(vx, va_fc, P_matrix)
                    pred_center = out[:, CONFIG["seq_len"]//2, :].argmax(dim=1)
                    y_center = vy[:, CONFIG["seq_len"]//2]
                    correct += (pred_center == y_center).sum().item()
                    v_total += y_center.size(0)
            
            val_acc = correct / v_total
            scheduler.step()
            
            if (epoch+1) % 5 == 0:
                print(f"  Epoch {epoch+1:02d} | Focal Loss: {avg_epoch_loss:.4f} | Acc: {val_acc:.4f}")
            
            early_stopping(val_acc, model)
            if early_stopping.early_stop: break

        #  训练结束，绘制该受试者的 Loss 曲线
        loss_plot_path = os.path.join(CONFIG["save_dir"], f"loss_curve_{test_sub}.png")
        plot_loss_curve(epoch_loss_history, loss_plot_path, test_sub)
        print(f">>> Loss 曲线已保存至: {loss_plot_path}")

        # 统计结果... (此处逻辑同原代码)
        model.load_state_dict(torch.load(save_path))
        model.eval()
        f_true, f_pred = [], []
        with torch.no_grad():
            for tx, ty, ta_fc in test_loader:
                tx, ty, ta_fc = tx.to(CONFIG["device"]), ty.to(CONFIG["device"]), ta_fc.to(CONFIG["device"])
                out, _ = model(tx, ta_fc, P_matrix)
                pred_center = out[:, CONFIG["seq_len"]//2, :].argmax(dim=1)
                y_center = ty[:, CONFIG["seq_len"]//2]
                f_true.extend(y_center.cpu().numpy())
                f_pred.extend(pred_center.cpu().numpy())
        
        subject_results.append({
            "Subject": test_sub,
            "Accuracy": accuracy_score(f_true, f_pred),
            "Macro-F1": f1_score(f_true, f_pred, average='macro'),
            "Kappa": cohen_kappa_score(f_true, f_pred)
        })
        global_y_true.extend(f_true)
        global_y_pred.extend(f_pred)
        writer.close()

    # 打印最终调试报告...
    df = pd.DataFrame(subject_results)
    print("\nDEBUG SUMMARY FOR 412 & 417:")
    print(df.to_string(index=False))
    plot_final_confusion_matrix(global_y_true, global_y_pred, os.path.join(CONFIG["save_dir"], "debug_cm_412_417.png"))

if __name__ == "__main__":
    train_loso()
