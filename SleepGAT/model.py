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
        self.lstm_drop = nn.Dropout(0.2)
        
        # 【跨 Epoch 记忆】：BiLSTM
        self.bilstm = nn.LSTM(hid_dim, hid_dim // 2, batch_first=True, bidirectional=True)
        self.classifier = nn.Linear(hid_dim, num_stages)

    def forward(self, x, A_fc, P_matrix):
        """
        x: (B, S, C, L)
        """
        B, S, C, L = x.shape
        x_flat = x.view(B * S, C, L)
        
        # 1. TCN
        feat_raw = self.tcn(x_flat.view(B*S*C, 1, L)) 
        feat = feat_raw.view(B * S, C, -1)
        
        # 2. MCGAT
        if A_fc.dim() == 3:
            A_fc = A_fc.unsqueeze(1).expand(B, S, C, C)
        out_m = self.mcgat(feat, A_fc.reshape(B*S, C, C))
        
        # 3. TRGAT
        out_t, s_logits = self.trgat(feat.mean(dim=1), P_matrix)
        
        # 4. Fusion
        fused = self.norm(self.fusion(out_m, out_t))
        
        # 5. 【核心修改】直接传入 BiLSTM，不传入且不返回 hidden
        seq_in = fused.view(B, S, -1)
        seq_in = self.lstm_drop(seq_in)
        lstm_out, _ = self.bilstm(seq_in) # PyTorch 会自动初始化隐状态为0
        
        # 此时 logits 包含了所有 S(21) 个时间步的预测，形状为 (B, S, 5)
        logits = self.classifier(lstm_out)
        
        return logits, s_logits


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
    def __init__(self, in_dim, out_dim, num_heads=4, num_channels=3, prior_lambda=1.0, prior_eps=1e-6):
        super(MCGAT, self).__init__()
        self.heads = num_heads
        self.head_dim = out_dim // num_heads
        self.num_channels = num_channels
        self.prior_lambda = prior_lambda
        self.prior_eps = prior_eps
        
        self.W = nn.Linear(in_dim, out_dim, bias=False)
        self.a = nn.Parameter(torch.Tensor(num_heads, 2 * self.head_dim))
        self.leakyrelu = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(0.3)
        
        # 【新增】维度对齐映射与 LayerNorm
        self.proj = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()
        self.norm = nn.LayerNorm(out_dim)
        
        self.channel_agg = nn.Sequential(
            nn.Linear(out_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )
        nn.init.xavier_uniform_(self.a)

    def forward(self, h, A_fc):
        BS, C, D = h.size()
        
        # 记录捷径分支 (BS, C, out_dim)
        residual = self.proj(h) 
        
        Wh = self.W(h).view(BS, C, self.heads, self.head_dim)
        Wh_i = Wh.unsqueeze(2).expand(BS, C, C, self.heads, self.head_dim)
        Wh_j = Wh.unsqueeze(1).expand(BS, C, C, self.heads, self.head_dim)
        e = torch.einsum('nijhd,hd->nijh', torch.cat([Wh_i, Wh_j], dim=-1), self.a)
        e = self.leakyrelu(e).permute(0, 3, 1, 2)
        
        A_prior = A_fc.unsqueeze(1).to(dtype=e.dtype).clamp_min(self.prior_eps)
        attention_logits = e + self.prior_lambda * torch.log(A_prior)
        attention = F.softmax(attention_logits, dim=-1)
        attention = self.dropout(attention)
        
        h_prime = torch.matmul(attention, Wh.permute(0, 2, 1, 3))
        h_prime = h_prime.permute(0, 2, 1, 3).contiguous().view(BS, C, -1)
        
        # 【新增】残差相加 + LayerNorm，稳住通道聚合前的特征分布
        h_prime = self.norm(h_prime + residual)
        
        w = F.softmax(self.channel_agg(h_prime), dim=1)
        out = torch.sum(h_prime * w, dim=1)
        return out
    
class TRGAT(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads=4, num_stages=5):
        super(TRGAT, self).__init__()
        self.num_stages = num_stages
        self.out_dim = out_dim
        
        self.stage_nodes = nn.Parameter(torch.randn(num_stages, out_dim))
        self.W_q = nn.Linear(in_dim, out_dim)
        self.W_k = nn.Linear(out_dim, out_dim)
        
        # 【新增】维度对齐与 LayerNorm
        self.proj = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x_epoch, P_matrix):
        BS = x_epoch.size(0)
        
        # 记录捷径分支 (BS, out_dim)
        residual = self.proj(x_epoch)
        
        updated_stages = torch.matmul(P_matrix, self.stage_nodes)
        
        query = self.W_q(x_epoch).unsqueeze(1)
        keys = self.W_k(updated_stages).unsqueeze(0)
        
        energy = torch.sum(query * keys, dim=-1) / (self.out_dim ** 0.5)
        s_probs = F.softmax(energy, dim=-1).unsqueeze(-1)
        
        out = torch.sum(s_probs * updated_stages.unsqueeze(0), dim=1)
        
        # 【新增】残差相加 + LayerNorm，确保本 Epoch 的特征不被彻底覆盖
        out = self.norm(out + residual)
        
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
