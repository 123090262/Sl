import torch
import torch.nn as nn
import torch.nn.functional as F

class SleepGATNet(nn.Module):
    def __init__(self, channel_names, fs=100, num_stages=5, feature_dim=128, hid_dim=256):
        super(SleepGATNet, self).__init__()
        self.C = len(channel_names)
        
        # 基础 TCN 提取
        self.tcn = EnhancedResTCN(fs=fs, feature_dim=feature_dim)
        
        # 双通道注意力
        self.mcgat = MCGAT(feature_dim, hid_dim, num_heads=4, num_channels=self.C)
        self.trgat = TRGAT(feature_dim, hid_dim, num_heads=4, num_stages=num_stages)
        
        self.norm = nn.LayerNorm(hid_dim)
        self.fusion = GatedFusion(hid_dim)
        
        # 【跨 Epoch 记忆】：BiLSTM
        self.bilstm = nn.LSTM(hid_dim, hid_dim // 2, batch_first=True, bidirectional=True)
        self.classifier = nn.Linear(hid_dim, num_stages)

    def forward(self, x, A_fc, P_matrix, hidden=None):
        """
        x: (B, S, C, L) -> (Batch, Seq, Chan, Len)
        A_fc: (B, S, C, C)
        P_matrix: (5, 5) 全局统计矩阵
        hidden: 传入 (h, c) 用于跨 Epoch 记忆
        """
        B, S, C, L = x.shape
        x_flat = x.view(B * S, C, L)
        
        # 1. TCN 提取各通道特征
        # 将 B*S*C 展平送入 TCN
        feat_raw = self.tcn(x_flat.view(B*S*C, 1, L)) 
        feat = feat_raw.view(B * S, C, -1) # (B*S, 3, 128)
        
        # 2. 分支一：MCGAT (维度从 3 变为 1)
        out_m = self.mcgat(feat, A_fc.view(B*S, C, C)) # (B*S, 256)
        
        # 3. 分支二：TRGAT (维度从 5 变为 1)
        # 输入取通道平均值作为 Epoch 初始表示
        out_t, s_logits = self.trgat(feat.mean(dim=1), P_matrix) # (B*S, 256)
        
        # 4. 融合
        fused = self.norm(self.fusion(out_m, out_t))
        
        # 5. 序列处理与跨 Epoch 记忆
        seq_in = fused.view(B, S, -1)
        # 如果是新的受试者，hidden 为 None；如果是连续序列，传入上一轮的 hidden
        lstm_out, new_hidden = self.bilstm(seq_in, hidden)
        
        # 取序列中间的 Epoch 作为分类结果 (或对整个序列分类)
        logits = self.classifier(lstm_out) # (B, S, 5)
        
        return logits, s_logits, new_hidden


class TCNResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, dilation, padding):
        super(TCNResidualBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, 
                               stride=stride, padding=padding, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        
        padding_same = kernel_size // 2 
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, 
                               stride=1, padding=padding_same, dilation=1)
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))
        shortcut = self.shortcut(x)
        
        # 规范化残差对齐：直接切片截断，保证相位不偏移，拒绝 pooling 强行对齐
        if shortcut.shape[-1] != out.shape[-1]:
            diff = shortcut.shape[-1] - out.shape[-1]
            if diff > 0:
                shortcut = shortcut[:, :, :-diff]
            else:
                out = out[:, :, :shortcut.shape[-1]]
                
        out += shortcut
        return self.relu(out)

class EnhancedResTCN(nn.Module):
    def __init__(self, fs, feature_dim=128): 
        super(EnhancedResTCN, self).__init__()
        k1 = int(50 * (fs / 100)) 
        s1 = int(5 * (fs / 100))
        
        self.layer1 = TCNResidualBlock(
            in_channels=1, out_channels=64, 
            kernel_size=k1, stride=s1, dilation=1, padding=k1//2
        )
        self.layer2 = TCNResidualBlock(
            in_channels=64, out_channels=128, 
            kernel_size=5, stride=2, dilation=2, padding=4 
        )
        self.layer3 = TCNResidualBlock(
            in_channels=128, out_channels=256, 
            kernel_size=3, stride=2, dilation=4, padding=4 
        )
        self.pool = nn.AdaptiveAvgPool1d(1) 
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, feature_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.pool(x)
        return self.fc(x)

class FeatureExtractor(nn.Module):
    def __init__(self, num_channels=3, fs=100, feature_dim=128):
        super(FeatureExtractor, self).__init__()
        self.C = num_channels
        self.fs = fs
        self.tcn = EnhancedResTCN(fs=self.fs, feature_dim=feature_dim)

    def forward(self, x):
        # x: (B*S, C, L) L 为完整的 3000 点
        B_S, C, L = x.size()
        x_batch_in = x.view(B_S * C, 1, L)
        features = self.tcn(x_batch_in)  # (B_S * C, feature_dim)
        out = features.view(B_S, C, -1)  # (B_S, C, feature_dim)
        return out
    
class MCGAT(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads=4, num_channels=3):
        super(MCGAT, self).__init__()
        self.heads = num_heads
        self.head_dim = out_dim // num_heads
        self.num_channels = num_channels
        
        self.W = nn.Linear(in_dim, out_dim, bias=False)
        self.a = nn.Parameter(torch.Tensor(num_heads, 2 * self.head_dim))
        self.leakyrelu = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(0.3)
        
        # 维度对齐核心：通道重要性聚合
        self.channel_agg = nn.Sequential(
            nn.Linear(out_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )
        
        nn.init.xavier_uniform_(self.a)

    def forward(self, h, A_fc):
        # h: (B*S, 3, in_dim), A_fc: (B*S, 3, 3)
        BS, C, D = h.size()
        Wh = self.W(h).view(BS, C, self.heads, self.head_dim) # (BS, 3, 4, D/4)
        
        # 构造 Attention Score
        Wh_i = Wh.unsqueeze(2).expand(BS, C, C, self.heads, self.head_dim)
        Wh_j = Wh.unsqueeze(1).expand(BS, C, C, self.heads, self.head_dim)
        e = torch.einsum('nijhd,hd->nijh', torch.cat([Wh_i, Wh_j], dim=-1), self.a)
        e = self.leakyrelu(e).permute(0, 3, 1, 2) # (BS, heads, 3, 3)
        
        # 融入 A_fc 先验：公式 alpha = Softmax(e * A_fc)
        A_prior = A_fc.unsqueeze(1) # (BS, 1, 3, 3)
        attention = F.softmax(e * A_prior, dim=-1)
        attention = self.dropout(attention)
        
        # 更新特征
        h_prime = torch.matmul(attention, Wh.permute(0, 2, 1, 3)) # (BS, heads, 3, head_dim)
        h_prime = h_prime.permute(0, 2, 1, 3).contiguous().view(BS, C, -1) # (BS, 3, out_dim)
        
        # 【对齐】：3个通道 -> 1个特征向量
        w = F.softmax(self.channel_agg(h_prime), dim=1) # (BS, 3, 1)
        out = torch.sum(h_prime * w, dim=1) # (BS, out_dim)
        return out
    
class TRGAT(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads=4, num_stages=5):
        super(TRGAT, self).__init__()
        self.num_stages = num_stages
        self.out_dim = out_dim
        
        # 5个全局睡眠阶段原型节点
        self.stage_nodes = nn.Parameter(torch.randn(num_stages, out_dim))
        self.W_q = nn.Linear(in_dim, out_dim)
        self.W_k = nn.Linear(out_dim, out_dim)
        
        # 阶段内 GAT 权重
        self.W_gat = nn.Linear(out_dim, out_dim)
        self.a_gat = nn.Parameter(torch.Tensor(num_heads, 2 * (out_dim//num_heads)))
        nn.init.xavier_uniform_(self.a_gat)

    def forward(self, x_epoch, P_matrix):
        # x_epoch: (BS, in_dim), P_matrix: (5, 5)
        BS = x_epoch.size(0)
        
        # 1. 阶段节点相互作用 (融入 P_matrix 转移先验)
        # 这里简化处理：直接利用 P_matrix 引导节点间特征流动
        updated_stages = torch.matmul(P_matrix, self.stage_nodes) # (5, out_dim)
        
        # 2. 【对齐】：当前特征与5个阶段节点做 Cross-Attention
        query = self.W_q(x_epoch).unsqueeze(1) # (BS, 1, out_dim)
        keys = self.W_k(updated_stages).unsqueeze(0) # (1, 5, out_dim)
        
        # 计算该 Epoch 对应各阶段的相似度 (辅助分类 Logits)
        energy = torch.sum(query * keys, dim=-1) / (self.out_dim ** 0.5) # (BS, 5)
        s_probs = F.softmax(energy, dim=-1).unsqueeze(-1) # (BS, 5, 1)
        
        # 根据概率重构特征：(BS, 5, 1) * (1, 5, out_dim) -> (BS, out_dim)
        out = torch.sum(s_probs * updated_stages.unsqueeze(0), dim=1)
        return out, energy
    
class GatedFusion(nn.Module):
    def __init__(self, dim):
        super(GatedFusion, self).__init__()
        self.fc = nn.Linear(dim * 2, dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x1, x2): 
        combined = torch.cat([x1, x2], dim=-1) 
        z = self.sigmoid(self.fc(combined)) 
        return z * x1 + (1 - z) * x2


class BiLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=1, dropout=0.2):
        super(BiLSTM, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.out_dim = hidden_dim * 2

    def forward(self, x):
        # x: (Batch, Seq_len, Input_dim) -> 此时的 Seq_len 为真实的时间步数 T(30)
        output, _ = self.lstm(x)
        return output