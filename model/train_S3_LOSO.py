import torch
import torch.nn as nn
import torch.optim as optim
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, cohen_kappa_score
from model import SleepGATNet
from ISRUCdataset import get_subject_data_dict, create_dataloader

# --- 配置参数 ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATASET_ROOT = "/mnt/sdc/sleep_project/dataset/S3"
CHANNELS = ["C3-A2", "C4-A1", "O1-A2", "O2-A1", "F4-A1", "F3-A2"]
BATCH_SIZE = 64
EPOCHS = 30
LR = 1e-3
SAVE_DIR = "./checkpoints_loso"
FS=200
os.makedirs(SAVE_DIR, exist_ok=True)

def plot_confusion_matrix(y_true, y_pred, title='Confusion Matrix', save_path=None):
    """
    绘制5×5混淆矩阵
    """
    # 类别标签
    class_names = ['W', 'N1', 'N2', 'N3', 'REM']
    
    # 计算混淆矩阵
    cm = confusion_matrix(y_true, y_pred, labels=range(len(class_names)))
    
    # 计算百分比（按行）
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    cm_percent = np.round(cm_percent, 2)
    
    # 创建图形
    plt.figure(figsize=(10, 8))
    
    # 使用seaborn热力图绘制
    ax = sns.heatmap(cm_percent, annot=True, fmt='.2f', cmap='Blues',
                     xticklabels=class_names, yticklabels=class_names,
                     cbar_kws={'label': 'Percentage (%)'})
    
    # 设置标签和标题
    plt.title(title, fontsize=16, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=14)
    plt.ylabel('True Label', fontsize=14)
    
    # 添加总数信息到每个格子
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            text = f"{cm_percent[i, j]:.2f}%\n({cm[i, j]})"
            ax.text(j + 0.5, i + 0.5, text,
                    ha='center', va='center',
                    color='black' if cm_percent[i, j] < 70 else 'white',
                    fontsize=10)
    
    plt.tight_layout()
    
    # 保存图像
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"混淆矩阵已保存到: {save_path}")
    
    plt.show()
    return cm

class ISFMR_Augmenter:
    """
    受试者内频率掩码-流形重构增强器 (无需修改模型)
    """
    def __init__(self, fs=100, n1_label=1):
        self.fs = fs
        self.n1_label = n1_label

    def __call__(self, x, y):
        """
        x: (B, C, 3000) 原始信号
        y: (B,) 标签
        """
        n1_indices = torch.where(y == self.n1_label)[0]
        if len(n1_indices) < 2:
            return x
        
        # 1. 转换到频域
        x_fft = torch.fft.rfft(x, dim=-1)
        amp = torch.abs(x_fft)
        phase = torch.angle(x_fft)
        
        # 2. 频率掩码：定义 Alpha(8-13Hz) 和 Theta(4-8Hz)
        freqs = torch.fft.rfftfreq(x.shape[-1], 1/self.fs)
        alpha_mask = (freqs >= 8) & (freqs <= 13)
        theta_mask = (freqs >= 4) & (freqs <= 8)

        # 3. 对 N1 样本进行受试者内流形重构
        n1_amp_avg = amp[n1_indices].mean(dim=0, keepdim=True)
        
        for idx in n1_indices:
            # 随机模拟不同的入睡状态
            mode = np.random.choice(['drop_alpha', 'boost_theta', 'mix_avg'])
            if mode == 'drop_alpha':
                amp[idx, :, alpha_mask] *= (0.5 + 0.5 * torch.rand(1).to(x.device))
            elif mode == 'boost_theta':
                amp[idx, :, theta_mask] *= (1.0 + 0.5 * torch.rand(1).to(x.device))
            
            # 流形重构
            beta = 0.1 + 0.2 * torch.rand(1).to(x.device)
            amp[idx] = (1 - beta) * amp[idx] + beta * n1_amp_avg.squeeze(0)

        # 4. 逆变换回时域
        x_aug_fft = amp * torch.exp(1j * phase)
        x_aug = torch.fft.irfft(x_aug_fft, n=x.shape[-1], dim=-1)
        
        return x_aug
    
def run_logo_fold(fold_idx, train_loader, test_loader):
    model = SleepGATNet(channel_names=CHANNELS, fs=FS).to(DEVICE)
    
    # 针对 N1 精度低，给 N1 (类别 1) 设置更高的权重
    class_weights = torch.tensor([1.0, 2.0, 1.0, 1.0, 1.0]).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    n1_augmenter = ISFMR_Augmenter(fs=FS, n1_label=1)

    for epoch in range(EPOCHS):
        model.train()
        # --- 修复 1: 确保在每个 epoch 开始时初始化 total_loss ---
        total_loss = 0 
        
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            
            # --- 修复 2: 这里的 n1_augmenter 现在是可调用的了 ---
            if epoch < int(EPOCHS * 0.8):
                x_in = n1_augmenter(x, y)
            else:
                x_in = x
            
            optimizer.zero_grad()
            logits, s_prob, _, _ = model(x_in)
            
            loss = criterion(logits, y) + 0.5 * criterion(s_prob, y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        if (epoch + 1) % 5 == 0:
            print(f"Fold {fold_idx} | Epoch {epoch+1}/{EPOCHS} | Loss: {total_loss/len(train_loader):.4f}")

    model.eval()
    y_true_fold, y_pred_fold = [], []
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            logits, _, _, _ = model(x)
            preds = torch.argmax(logits, dim=1)
            y_true_fold.extend(y.cpu().numpy())
            y_pred_fold.extend(preds.cpu().numpy())
            
    return np.array(y_true_fold), np.array(y_pred_fold)

if __name__ == "__main__":
    subjects = [f"{i}" for i in range(1, 11)]
    all_data = get_subject_data_dict(DATASET_ROOT, subjects, CHANNELS)
    
    all_y_true = []
    all_y_pred = []
    
    # 存储每个fold的结果用于分析
    fold_results = []

    # 10轮 LOSO 循环
    for i in range(len(subjects)):
        test_sub = [subjects[i]]
        train_subs = [s for s in subjects if s != subjects[i]]
        
        print(f"\n>>>> Round {i+1}/10 | Testing on Subject: {test_sub}")
        
        train_loader = create_dataloader(all_data, train_subs, batch_size=BATCH_SIZE)
        test_loader = create_dataloader(all_data, test_sub, batch_size=BATCH_SIZE, shuffle=False)
        
        # 执行训练与测试
        y_true, y_pred = run_logo_fold(i+1, train_loader, test_loader)
        
        all_y_true.extend(y_true)
        all_y_pred.extend(y_pred)
        
        # 保存每个fold的结果
        fold_results.append({
            'fold': i+1,
            'test_subject': test_sub[0],
            'y_true': y_true,
            'y_pred': y_pred
        })

    # --- 最终聚合评估 ---
    print("\n" + "="*60)
    print("FINAL LOSO RESULTS (Aggregated)")
    print("="*60)
    
    # 计算整体指标
    target_names = ['W', 'N1', 'N2', 'N3', 'REM']
    print("\n分类报告:")
    print(classification_report(all_y_true, all_y_pred, target_names=target_names, digits=4))
    
    kappa = cohen_kappa_score(all_y_true, all_y_pred)
    print(f"\nOverall Cohen's Kappa: {kappa:.4f}")
    
    # 计算并显示总体准确率
    accuracy = np.mean(np.array(all_y_true) == np.array(all_y_pred))
    print(f"Overall Accuracy: {accuracy:.4f}")

    # 绘制总体混淆矩阵
    print("\n绘制总体混淆矩阵...")
    overall_cm = plot_confusion_matrix(
        all_y_true, all_y_pred,
        title='Overall Confusion Matrix (LOSO)',
        save_path='overall_confusion_matrix.png'
    )
    
    # 可选：绘制每个fold的混淆矩阵
    print("\n绘制每个fold的混淆矩阵...")
    os.makedirs('fold_confusion_matrices', exist_ok=True)
    
    for i, fold_result in enumerate(fold_results):
        fold_cm = plot_confusion_matrix(
            fold_result['y_true'], fold_result['y_pred'],
            title=f'Fold {fold_result["fold"]} - Subject {fold_result["test_subject"]}',
            save_path=f'fold_confusion_matrices/fold_{fold_result["fold"]}_subject_{fold_result["test_subject"]}.png'
        )
        plt.close()  # 关闭图形以免显示所有fold的图
    
    # 保存结果数据
    np.savez("loso_final_results.npz", 
             true=all_y_true, 
             pred=all_y_pred,
             fold_results=fold_results)
    
    print("\n" + "="*60)
    print("所有分析完成！")
    print("保存的文件:")
    print("1. loso_final_results.npz - 所有预测结果")
    print("2. overall_confusion_matrix.png - 总体混淆矩阵")
    print("3. fold_confusion_matrices/ - 每个fold的混淆矩阵")
    print("="*60)