import matplotlib
import mne
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import copy
import os
import numpy as np
import seaborn as sns

from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import confusion_matrix, classification_report, cohen_kappa_score, roc_auc_score

# 导入你定义的模型和数据处理脚本
from model import SleepGATNet 
from SleepEDFdataset import get_subject_data_dict, create_dataloader

# --- 1. 配置参数 ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATASET_ROOT = r"E:\EEG\dataset\sleep-edf-database-expanded-1.0.0\SleepEDF-20"
CHANNELS = ["EEG Fpz-Cz", "EEG Pz-Oz"] 
FS = 100  
BATCH_SIZE = 64
EPOCHS = 50
LR = 1e-4
SAVE_DIR = "./checkpoints_edf_loso"
LOG_DIR = os.path.join(SAVE_DIR, "tensorboard_logs")
os.makedirs(SAVE_DIR, exist_ok=True)

# --- 2. 辅助函数 ---
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
    if save_path: plt.savefig(save_path, dpi=300)
    plt.close()

# --- 3. 核心训练函数 ---
def run_loso_fold(fold_idx, train_loader, test_loader, sub_name):
    # 为当前受试者创建 TensorBoard 记录器
    writer = SummaryWriter(log_dir=os.path.join(LOG_DIR, sub_name))
    
    model = SleepGATNet(channel_names=CHANNELS, fs=FS).to(DEVICE)

    # 动态计算类别权重以应对数据不平衡
    num_classes = 5
    class_counts = np.zeros(num_classes, dtype=np.float32)
    for _, y_batch, _ in train_loader:
        class_counts += np.bincount(y_batch.numpy().astype(np.int64), minlength=num_classes)
    
    class_weights = 1.0 / (class_counts / class_counts.sum() + 1e-9)
    class_weights = torch.tensor(class_weights / class_weights.mean(), dtype=torch.float32, device=DEVICE)
    
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-3) # 增大 weight_decay 抑制过拟合
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    best_val_loss = float('inf')
    best_state = None
    patience, no_improve = 10, 0
    global_step = 0

    for epoch in range(EPOCHS):
        # --- 训练阶段 ---
        model.train()
        train_loss, correct_train, total_train = 0.0, 0, 0
        
        for x, y, a_plv in train_loader:
            x, y, a_plv = x.to(DEVICE), y.to(DEVICE), a_plv.to(DEVICE)
            optimizer.zero_grad()
            
            logits, _ = model(x, a_plv)  # SleepGATNet 返回 (logits, s_logits)
            loss = criterion(logits, y)
            
            if torch.isnan(loss):
                print(f"Warning: NaN loss detected in {sub_name}, skipping batch.")
                continue
                
            loss.backward()
            
            # 监控并防止梯度爆炸：梯度裁剪
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            writer.add_scalar('Train/Gradient_Norm', grad_norm, global_step)
            writer.add_scalar('Train/Batch_Loss', loss.item(), global_step)
            
            optimizer.step()
            
            train_loss += loss.item()
            correct_train += (torch.argmax(logits, dim=1) == y).sum().item()
            total_train += y.size(0)
            global_step += 1
            
        # --- 验证阶段 ---
        model.eval()
        val_loss, correct_val, total_val = 0.0, 0, 0
        with torch.no_grad():
            for x, y, a_plv in test_loader:
                x, y, a_plv = x.to(DEVICE), y.to(DEVICE), a_plv.to(DEVICE)
                logits, _ = model(x, a_plv)
                val_loss += criterion(logits, y).item()
                correct_val += (torch.argmax(logits, dim=1) == y).sum().item()
                total_val += y.size(0)

        # 计算指标
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(test_loader)
        train_acc = correct_train / total_train
        val_acc = correct_val / total_val

        # 将指标写入 TensorBoard 观察过拟合
        writer.add_scalar('Epoch/Loss_Train', avg_train_loss, epoch)
        writer.add_scalar('Epoch/Loss_Val', avg_val_loss, epoch)
        writer.add_scalar('Epoch/Acc_Train', train_acc, epoch)
        writer.add_scalar('Epoch/Acc_Val', val_acc, epoch)
        writer.add_scalar('Train/Learning_Rate', optimizer.param_groups[0]['lr'], epoch)

        scheduler.step(avg_val_loss)

        if avg_val_loss < best_val_loss - 1e-4:
            best_val_loss, best_state, no_improve = avg_val_loss, copy.deepcopy(model.state_dict()), 0
        else:
            no_improve += 1

        if (epoch + 1) % 5 == 0:
            print(f"Sub {sub_name} | Ep {epoch+1} | Train Loss: {avg_train_loss:.3f} | Val Acc: {val_acc:.4f}")

        if no_improve >= patience: break

    if best_state: model.load_state_dict(best_state)
    
    # 最终测试获取预测结果
    y_true_f, y_pred_f, y_prob_f = [], [], []
    model.eval()
    with torch.no_grad():
        for x, y, a_plv in test_loader:
            x, y, a_plv = x.to(DEVICE), y.to(DEVICE), a_plv.to(DEVICE)
            logits, _ = model(x, a_plv)
            y_true_f.extend(y.cpu().numpy())
            y_pred_f.extend(torch.argmax(logits, dim=1).cpu().numpy())
            y_prob_f.extend(torch.softmax(logits, dim=1).cpu().numpy())

    writer.close()
    return np.array(y_true_f), np.array(y_pred_f), np.array(y_prob_f), model

# --- 4. 主程序 ---
if __name__ == "__main__":
    subjects = [f"SC4{i:02d}" for i in range(20)] 
    # 此处调用需配合你在上一步修改过的、支持受试者级归一化的 get_subject_data_dict
    all_data = get_subject_data_dict(DATASET_ROOT, subjects, CHANNELS)

    all_y_true, all_y_pred, all_y_prob = [], [], []

    for i, test_sub in enumerate(subjects):
        train_subs = [s for s in subjects if s != test_sub]
        print(f"\n>>>> Fold {i+1}/20 | Test Subject: {test_sub}")
        
        train_loader = create_dataloader(all_data, train_subs, batch_size=BATCH_SIZE)
        test_loader = create_dataloader(all_data, [test_sub], batch_size=BATCH_SIZE, shuffle=False)
        
        y_true, y_pred, y_prob, trained_model = run_loso_fold(i+1, train_loader, test_loader, test_sub)
        
        torch.save(trained_model.state_dict(), os.path.join(SAVE_DIR, f"model_{test_sub}.pth"))
        
        all_y_true.extend(y_true); all_y_pred.extend(y_pred); all_y_prob.extend(y_prob)

    # 最终统计
    print("\n" + "="*30 + " FINAL RESULTS " + "="*30)
    print(classification_report(all_y_true, all_y_pred, target_names=['W', 'N1', 'N2', 'N3', 'REM'], digits=4))
    plot_confusion_matrix(all_y_true, all_y_pred, save_path=f'{SAVE_DIR}/final_cm.png')