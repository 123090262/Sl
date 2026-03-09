import matplotlib
matplotlib.use('Agg') # 【新增】强制使用无头模式，防止服务器跑一半因为没屏幕而报错中断！
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import os
import numpy as np
import json
from sklearn.metrics import confusion_matrix, classification_report, cohen_kappa_score, roc_auc_score
from model import SleepGATNet 
from SleepEDFdataset import get_subject_data_dict, create_dataloader
import seaborn as sns

# --- 1. 配置参数 ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATASET_ROOT = "/mnt/sdc/sleep_project/dataset/SleepEDF-20"
CHANNELS =["EEG Fpz-Cz", "EEG Pz-Oz"] 
FS = 100  
BATCH_SIZE = 64
EPOCHS = 30
LR = 1e-3
SAVE_DIR = "./checkpoints_edf_logo"
os.makedirs(SAVE_DIR, exist_ok=True)

# --- 2.混淆矩阵绘制 ---
def plot_confusion_matrix(y_true, y_pred, title='Confusion Matrix', save_path=None):
    class_names =['W', 'N1', 'N2', 'N3', 'REM']
    cm = confusion_matrix(y_true, y_pred, labels=range(len(class_names)))
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    
    plt.figure(figsize=(10, 8))
    ax = sns.heatmap(cm_percent, annot=True, fmt='.2f', cmap='Greens',
                     xticklabels=class_names, yticklabels=class_names)
    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.close() # 【修改】关闭图片而不是show，防止内存泄漏和报错

# --- 3. LOGO 训练逻辑 ---
class ISFMR_Augmenter:
    def __init__(self, fs=100, n1_label=1):
        self.fs = fs
        self.n1_label = n1_label

    def __call__(self, x, y):
        n1_indices = torch.where(y == self.n1_label)[0]
        if len(n1_indices) < 2: return x
        
        x_fft = torch.fft.rfft(x, dim=-1)
        amp, phase = torch.abs(x_fft), torch.angle(x_fft)
        
        freqs = torch.fft.rfftfreq(x.shape[-1], 1/self.fs)
        alpha_mask = (freqs >= 8) & (freqs <= 13)
        theta_mask = (freqs >= 4) & (freqs <= 8)

        n1_amp_avg = amp[n1_indices].mean(dim=0, keepdim=True)
        
        for idx in n1_indices:
            mode = np.random.choice(['drop_alpha', 'boost_theta', 'mix_avg'])
            if mode == 'drop_alpha':
                amp[idx, :, alpha_mask] *= (0.5 + 0.5 * torch.rand(1).to(x.device))
            elif mode == 'boost_theta':
                amp[idx, :, theta_mask] *= (1.0 + 0.5 * torch.rand(1).to(x.device))
            
            beta = 0.1 + 0.2 * torch.rand(1).to(x.device)
            amp[idx] = (1 - beta) * amp[idx] + beta * n1_amp_avg.squeeze(0)

        x_aug_fft = amp * torch.exp(1j * phase)
        return torch.fft.irfft(x_aug_fft, n=x.shape[-1], dim=-1)
    
def run_logo_fold(fold_idx, train_loader, test_loader):
    model = SleepGATNet(channel_names=CHANNELS, fs=FS).to(DEVICE)
    class_weights = torch.tensor([1.0, 2.0, 1.0, 1.0, 1.0]).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    n1_augmenter = ISFMR_Augmenter(fs=FS, n1_label=1)

    history = {'train_loss':[], 'train_acc': [], 'val_loss': [], 'val_acc':[]}

    for epoch in range(EPOCHS):
        model.train()
        train_loss, correct_train, total_train = 0.0, 0, 0
        
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            x_in = n1_augmenter(x, y) if epoch < int(EPOCHS * 0.8) else x
            
            optimizer.zero_grad()
            logits, s_prob, _, _ = model(x_in)
            loss = criterion(logits, y) + 0.5 * criterion(s_prob, y)
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
                loss_val = criterion(logits, y)
                val_loss += loss_val.item()
                preds = torch.argmax(logits, dim=1)
                correct_val += (preds == y).sum().item()
                total_val += y.size(0)

        history['train_loss'].append(float(train_loss / len(train_loader)))
        history['train_acc'].append(float(correct_train / total_train))
        history['val_loss'].append(float(val_loss / len(test_loader)))
        history['val_acc'].append(float(correct_val / total_val))
        
        if (epoch + 1) % 5 == 0:
            print(f"Fold {fold_idx} | Epoch {epoch+1}/{EPOCHS} | "
                  f"Train Loss: {history['train_loss'][-1]:.4f}, Acc: {history['train_acc'][-1]:.4f} | "
                  f"Val Loss: {history['val_loss'][-1]:.4f}, Acc: {history['val_acc'][-1]:.4f}")

    model.eval()
    y_true_fold, y_pred_fold, y_prob_fold = [], [],[]
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            logits, _, _, _ = model(x)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)
            
            y_true_fold.extend(y.cpu().numpy())
            y_pred_fold.extend(preds.cpu().numpy())
            y_prob_fold.extend(probs.cpu().numpy())
            
    # 【新增】返回 model，以便在主循环中保存权重
    return np.array(y_true_fold), np.array(y_pred_fold), np.array(y_prob_fold), history, model

if __name__ == "__main__":
    subjects =[f"SC4{i:02d}" for i in range(20)] 
    all_data = get_subject_data_dict(DATASET_ROOT, subjects, CHANNELS)
    groups = [subjects[i:i+2] for i in range(0, 20, 2)]

    all_y_true, all_y_pred, all_y_prob = [], [], []
    all_history =[] 

    for i in range(len(groups)):
        test_subs = groups[i]
        train_subs =[s for j, group in enumerate(groups) if i != j for s in group]
        
        print(f"\n>>>> Fold {i+1}/10 | Testing on Groups: {test_subs}")
        train_loader = create_dataloader(all_data, train_subs, batch_size=BATCH_SIZE)
        test_loader = create_dataloader(all_data, test_subs, batch_size=BATCH_SIZE, shuffle=False)
        
        # 接收训练好的模型
        y_true, y_pred, y_prob, history, trained_model = run_logo_fold(i+1, train_loader, test_loader)
        
        # 【新增】保存这一折的模型权重，供日后画 CAM/t-SNE 使用
        model_save_path = os.path.join(SAVE_DIR, f"edf20_model_fold_{i+1}.pth")
        torch.save(trained_model.state_dict(), model_save_path)
        print(f"[*] 模型权重已保存至: {model_save_path}")
        
        all_y_true.extend(y_true)
        all_y_pred.extend(y_pred)
        all_y_prob.extend(y_prob)
        all_history.append(history)

    print("\n" + "="*60)
    print("FINAL LOGO RESULTS (Sleep-EDF 20 Subs)")
    print(classification_report(all_y_true, all_y_pred, target_names=['W', 'N1', 'N2', 'N3', 'REM'], digits=4))
    
    macro_auroc = roc_auc_score(all_y_true, all_y_prob, multi_class='ovo')
    print(f"Overall Macro AUROC: {macro_auroc:.4f}")
    
    plot_confusion_matrix(all_y_true, all_y_pred, title='Overall Sleep-EDF LOGO CM', save_path=f'{SAVE_DIR}/edf_logo_cm.png')
    
    np.savez(f"{SAVE_DIR}/edf20_predictions.npz", y_true=all_y_true, y_pred=all_y_pred, y_prob=all_y_prob)
    with open(f"{SAVE_DIR}/edf20_training_history.json", "w") as f:
        json.dump(all_history, f)
        
    print(f"\n[全部完成] 数据和模型均已妥善保存于 {SAVE_DIR} 目录！")
# --- END OF FILE ---