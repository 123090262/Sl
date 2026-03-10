import torch
import torch.nn as nn
import torch.nn.functional as F

class SleepGATNet(nn.Module):
    def __init__(self, channel_names, fs=100, num_stages=5, feature_dim=128, hid_dim=256):
        super(SleepGATNet, self).__init__()
        self.C = len(channel_names)
        self.fs = fs
        
        # 1. 结构先验构建器
        self.prior_builder = PriorMatrixBuilder(channel_names=channel_names)
        
        # 2. 浅层 TCN 特征提取器
        self.feature_extractor = FeatureExtractor(
            num_channels=self.C, 
            fs=self.fs,
            feature_dim=feature_dim
        )
        
        # 3. 双路图注意力网络 (处理空间特征)
        self.tagat = TAGAT(
            in_dim=feature_dim, 
            hid_dim=hid_dim, 
            num_nodes=self.C
        )
        
        self.scgat = SCGAT(
            in_channels=feature_dim, 
            out_channels=hid_dim, 
            stage_dim=64, 
            num_stages=num_stages
        )
        
        # 4. 门控融合
        self.gate_fusion = GatedFusion(dim=hid_dim)
        
        # 5. 通道注意力 (空间聚合)
        self.channel_att = ChannelAttention(dim=hid_dim)
        
        # 6. 时间序列建模 (处理 30 个时间窗口)
        self.bilstm = BiLSTM(input_dim=hid_dim, hidden_dim=hid_dim)
        
        # 7. 分类头 (BiLSTM 输出维度为 hid_dim * 2)
        self.classifier = nn.Sequential(
            nn.Linear(hid_dim * 2, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_stages)
        )

    def forward(self, x_raw):
        """
        Input x_raw: (B, C, 30*fs) -> 30s 信号，每通道长度应为 30*采样率
        """
        # Step 1: 构建先验邻接矩阵 A_init (C, C)
        A_init = self.prior_builder(x_raw) 
        
        # Step 2: 提取局部时空特征
        x_feat = self.feature_extractor(x_raw) # (B, C, T=30, feature_dim)
        B, C, T, F_dim = x_feat.shape
        
        # 将 Batch 和 Time 维度合并，方便 GAT 在每个时间步独立处理空间图
        # 转置为 (B, T, C, F_dim) -> reshape (B*T, C, F_dim)
        x_spatial = x_feat.transpose(1, 2).reshape(B * T, C, F_dim)
        
        # Step 3: 并行执行 TAGAT 与 SCGAT
        h_tagat = self.tagat(x_spatial) # (B*T, C, hid_dim)
        h_scgat, s_logits_bt = self.scgat(x_spatial, A_init) # (B*T, C, hid_dim), (B*T, num_stages)

        # 对各个时间步的 stage 预测取平均作为辅助输出（logits）
        s_logits = s_logits_bt.view(B, T, -1).mean(dim=1) # (B, num_stages)
        
        # Step 4: 门控融合
        h_fused = self.gate_fusion(h_tagat, h_scgat) # (B*T, C, hid_dim)
        
        # Step 5: 通道注意力与空间聚合
        # 为每个通道分配权重，并聚合为一个时间步的特征表示
        h_att, att_weights = self.channel_att(h_fused) # h_att: (B*T, C, hid_dim)
        h_space_pooled = torch.sum(h_att, dim=1)       # (B*T, hid_dim) - 空间节点收缩
        
        # 恢复时间序列维度 (B, T, hid_dim)
        h_seq = h_space_pooled.view(B, T, -1)
        
        # Step 6: BiLSTM 建模时间序列间的依赖 (真正的时序建模)
        h_lstm = self.bilstm(h_seq) # (B, T, hid_dim * 2)
        
        # Step 7: 全局时序平均池化 (GAP) 与 分类
        h_gap = torch.mean(h_lstm, dim=1) # (B, hid_dim * 2)
        logits = self.classifier(h_gap)   # (B, num_stages)
        
        # 改变 att_weights 形状以供外部可视化分析 (B, T, C, 1)
        att_weights = att_weights.view(B, T, C, 1)
        
        return logits, s_logits, A_init, att_weights


class TCNResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, dilation, padding):
        super(TCNResidualBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, 
                               stride=stride, padding=padding, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        # conv2 保持 stride=1，使用显式 padding 替代 'same' 避免偶数 kernel 的警告
        padding_same = kernel_size // 2  # 与 PyTorch same 行为一致，偶数 kernel 时偏大保证不截断
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
        # 主路径与 shortcut 可能因 stride/kernel 计算差异导致长度不同，需对齐
        if shortcut.shape[-1] != out.shape[-1]:
            shortcut = F.adaptive_avg_pool1d(shortcut, out.shape[-1])
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
    def __init__(self, num_channels=6, fs=200, time_steps=30, feature_dim=128):
        super(FeatureExtractor, self).__init__()
        self.C = num_channels
        self.fs = fs
        self.T_prime = time_steps 
        self.tcn = EnhancedResTCN(fs=self.fs, feature_dim=feature_dim)

    def forward(self, x):
        B, C, T_total = x.size()
        if T_total != self.T_prime * self.fs:
            raise ValueError(f"输入数据长度 {T_total} 与设定 {self.T_prime}*{self.fs} 不匹配")
            
        x_reshaped = x.view(B, C, self.T_prime, self.fs)
        x_batch_in = x_reshaped.view(B * C * self.T_prime, 1, self.fs)
        features = self.tcn(x_batch_in)
        out = features.view(B, C, self.T_prime, -1) #1
        return out
    
class PriorMatrixBuilder(nn.Module):
    def __init__(self, channel_names, sigma=1.0, beta=10.0, tau=0.5):
        super(PriorMatrixBuilder, self).__init__()
        self.C = len(channel_names)
        self.sigma = sigma
        self.beta = beta
        self.tau = tau
        
        coords = {
            "C3-A2": [-0.707, 0.0, 0.707],  "C4-A1": [0.707, 0.0, 0.707],
            "O1-A2": [-0.4, -0.9, 0.1],     "O2-A1": [0.4, -0.9, 0.1],
            "F3-A2": [-0.4, 0.8, 0.4],      "F4-A1": [0.4, 0.8, 0.4],
            "EEG Fpz-Cz":[0.0, 0.95, 0.2],  "EEG Pz-Oz":[0.0,-0.95,0.2]  
        }
        # 如果有通道名不在 coords 中，需添加容错机制
        pos_list = [coords.get(name, [0.0, 0.0, 0.0]) for name in channel_names]
        pos = torch.tensor(pos_list, dtype=torch.float32) 
        
        dist = torch.cdist(pos, pos, p=2) 
        self.register_buffer('A_geo', torch.exp(-(dist**2) / (self.sigma**2))) 
        
        self.weight_fusion = nn.Parameter(torch.tensor([0.5, 0.5])) 
        self.E = nn.Parameter(torch.randn(self.C, self.C) * 0.01) 

    def compute_plv(self, x_raw):
        B, C, T = x_raw.shape
        Xf = torch.fft.fft(x_raw, dim=-1)
        h = torch.zeros(T, device=x_raw.device) 
        if T % 2 == 0: 
            h[0] = h[T//2] = 1 
            h[1:T//2] = 2 
        else:
            h[0] = 1; h[1:(T+1)//2] = 2
        
        analytic_signal = torch.fft.ifft(Xf * h, dim=-1)
        phase = torch.angle(analytic_signal) 
        
        phase_diff = phase.unsqueeze(2) - phase.unsqueeze(1) 
        real_avg = torch.cos(phase_diff).mean(dim=-1)
        imag_avg = torch.sin(phase_diff).mean(dim=-1) 
        plv_matrix = torch.sqrt(real_avg**2 + imag_avg**2) 
        return plv_matrix 

    def forward(self, x_raw):
        A_plv = self.compute_plv(x_raw).mean(dim=0) 
        w = torch.softmax(self.weight_fusion, dim=0)
        A_0 = w[0] * self.A_geo + w[1] * A_plv
        
        A_init = A_0 * F.softplus(self.E) 
        A_init = (A_init + A_init.T) / 2 
        
        # 使用 mask 替换 in-place 操作，防止反向传播报错
        mask = 1.0 - torch.eye(self.C, device=A_init.device)
        A_init = A_init * mask 
        
        A_init = torch.sigmoid(self.beta * (A_init - self.tau))
        return A_init

class SCGAT(nn.Module):
    def __init__(self, in_channels, out_channels, stage_dim=64, num_stages=5, num_heads=8):
        super(SCGAT, self).__init__()
        self.num_heads = num_heads
        self.d_k = out_channels // num_heads
        
        self.lin_project = nn.Linear(in_channels, out_channels)
        self.stage_mlp = nn.Sequential(
            nn.Linear(in_channels, 128), nn.ReLU(),
            nn.Linear(128, num_stages)
        )
        self.mlp_c = nn.Sequential(
            nn.Linear(num_stages, stage_dim), nn.ReLU(), 
            nn.Linear(stage_dim, stage_dim)
        )
        
        # 改用更安全的 Linear 计算注意力分数，替代复杂的 MatMul 广播
        self.att_scorer = nn.Linear(2 * self.d_k + stage_dim, 1, bias=False)
        self.lambda_prior = nn.Parameter(torch.ones(num_heads))

    def forward(self, x, A_prior):
        B, C, _ = x.shape
        x_proj = self.lin_project(x).view(B, C, self.num_heads, self.d_k)
        
        s_logits = self.stage_mlp(x.mean(dim=1))
        s_prob = F.softmax(s_logits, dim=-1)
        c = self.mlp_c(s_prob) # (B, stage_dim)
        
        # 显式展开维度以规避广播错误 (B, 目标节点, 源节点, 头, 特征)
        x_dst = x_proj.unsqueeze(2).expand(-1, -1, C, -1, -1) 
        x_src = x_proj.unsqueeze(1).expand(-1, C, -1, -1, -1) 
        c_rep = c.view(B, 1, 1, 1, -1).expand(-1, C, C, self.num_heads, -1)
        
        att_input = torch.cat([x_src, x_dst, c_rep], dim=-1) # (B, C, C, H, 2*d_k + stage_dim)
        e = F.leaky_relu(self.att_scorer(att_input).squeeze(-1), 0.2) # (B, C, C, H)
        
        # 融合先验矩阵
        prior_term = self.lambda_prior.view(1, 1, 1, -1) * torch.log(A_prior.unsqueeze(0).unsqueeze(-1) + 1e-9)
        combined_e = e + prior_term
        
        alpha = F.softmax(combined_e, dim=2) # 沿源节点(dim=2)进行 Softmax
        
        # 注意力加权聚合
        alpha = alpha.unsqueeze(-1) # (B, C, C, H, 1)
        x_val = x_proj.unsqueeze(1).expand(-1, C, -1, -1, -1) # (B, C, C, H, d_k)
        out = (alpha * x_val).sum(dim=2) # 沿源节点求和 -> (B, C, H, d_k)
        
        return out.reshape(B, C, -1), s_logits
    
class TAGAT(nn.Module):
    def __init__(self, in_dim, hid_dim, num_nodes, t_enc_dim=32, num_heads=4):
        super(TAGAT, self).__init__()
        self.num_heads = num_heads
        self.d_k = hid_dim // num_heads
        self.lin_project = nn.Linear(in_dim, hid_dim)
        self.node_time_pos = nn.Parameter(torch.randn(num_nodes, 1))
        self.time_mlp = nn.Sequential(nn.Linear(1, t_enc_dim), nn.ReLU(), nn.Linear(t_enc_dim, t_enc_dim))
        
        self.att_scorer = nn.Linear(2 * self.d_k + t_enc_dim, 1, bias=False)

    def forward(self, x):
        B, C, _ = x.shape
        x_proj = self.lin_project(x).view(B, C, self.num_heads, self.d_k)
        
        tau = self.node_time_pos.unsqueeze(1) - self.node_time_pos.unsqueeze(0) # (C, C, 1)
        t_enc = self.time_mlp(tau) # (C, C, t_enc_dim)
        
        x_dst = x_proj.unsqueeze(2).expand(-1, -1, C, -1, -1) 
        x_src = x_proj.unsqueeze(1).expand(-1, C, -1, -1, -1) 
        t_enc_rep = t_enc.unsqueeze(0).unsqueeze(3).expand(B, -1, -1, self.num_heads, -1)
        
        att_input = torch.cat([x_src, x_dst, t_enc_rep], dim=-1)
        e = F.leaky_relu(self.att_scorer(att_input).squeeze(-1), 0.2)
        
        alpha = F.softmax(e, dim=2)
        
        alpha = alpha.unsqueeze(-1) 
        x_val = x_proj.unsqueeze(1).expand(-1, C, -1, -1, -1) 
        out = (alpha * x_val).sum(dim=2)
        
        return out.reshape(B, C, -1)
    
class GatedFusion(nn.Module):
    def __init__(self, dim):
        super(GatedFusion, self).__init__()
        self.fc = nn.Linear(dim * 2, dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x1, x2): 
        combined = torch.cat([x1, x2], dim=-1) 
        z = self.sigmoid(self.fc(combined)) 
        return z * x1 + (1 - z) * x2

class ChannelAttention(nn.Module):
    def __init__(self, dim):
        super(ChannelAttention, self).__init__()
        self.att = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.ReLU(),
            nn.Linear(dim // 2, 1),
            nn.Softmax(dim=1) # 在通道维度进行 Softmax
        )

    def forward(self, x):
        # x: (B*T, C, D)
        weights = self.att(x) # (B*T, C, 1)
        return x * weights, weights

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