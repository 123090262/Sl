import torch
import torch.nn as nn
import torch.nn.functional as F

class SleepGATNet(nn.Module):
    def __init__(self, channel_names, fs=100, num_stages=5, feature_dim=128, hid_dim=256, seq_len=5):
        super(SleepGATNet, self).__init__()
        self.C = len(channel_names)
        self.fs = fs
        
        self.prior_builder = PriorMatrixBuilder(channel_names=channel_names)
        self.feature_extractor = FeatureExtractor(num_channels=self.C, fs=self.fs, feature_dim=feature_dim)
        
        # 创新点保留
        self.tagat = TAGAT(feature_dim, hid_dim, self.C)
        self.scgat = SCGAT(feature_dim, hid_dim, num_stages=num_stages)
        
        # 加入 LayerNorm 增加数值稳定性，防止 GAT 梯度爆炸
        self.norm_t = nn.LayerNorm(hid_dim)
        self.norm_s = nn.LayerNorm(hid_dim)
        
        self.gate_fusion = GatedFusion(dim=hid_dim)
        self.context_bilstm = nn.LSTM(hid_dim, hid_dim//2, batch_first=True, bidirectional=True)
        self.classifier = nn.Linear(hid_dim, num_stages)

    def forward(self, x, a_plv):
        B, S, C, L = x.shape
        x_flat = x.view(B * S, C, L)
        a_plv_flat = a_plv.view(B * S, C, C)

        A_prior = self.prior_builder(a_plv_flat)
        feat = self.feature_extractor(x_flat) # (B*S, C, D)

        # GAT 分支 + 稳定性 Norm
        tagat_out = self.norm_t(self.tagat(feat))              
        scgat_out, s_logits = self.scgat(feat, A_prior)
        scgat_out = self.norm_s(scgat_out)

        fused = self.gate_fusion(tagat_out.mean(1), scgat_out.mean(1))
        
        seq_feat = fused.view(B, S, -1)
        out, _ = self.context_bilstm(seq_feat)
        
        logits = self.classifier(out[:, S//2, :])
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
    def __init__(self, num_channels=6, fs=100, feature_dim=128):
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
        pos_list = [coords.get(name, [0.0, 0.0, 0.0]) for name in channel_names]
        pos = torch.tensor(pos_list, dtype=torch.float32) 
        
        dist = torch.cdist(pos, pos, p=2) 
        self.register_buffer('A_geo', torch.exp(-(dist**2) / (self.sigma**2))) 
        
        self.weight_fusion = nn.Parameter(torch.tensor([0.5, 0.5])) 
        self.E = nn.Parameter(torch.randn(self.C, self.C) * 0.01) 

    def forward(self, A_plv):
        # A_plv: (B*S, C, C) 取 batch 平均以获得稳定的先验
        A_plv_mean = A_plv.mean(dim=0)
        
        w = torch.softmax(self.weight_fusion, dim=0)
        A_0 = w[0] * self.A_geo + w[1] * A_plv_mean
        
        A_init = A_0 * F.softplus(self.E) 
        A_init = (A_init + A_init.T) / 2 
        
        mask = 1.0 - torch.eye(self.C, device=A_init.device)
        A_init = A_init * mask 
        
        A_init = torch.sigmoid(self.beta * (A_init - self.tau))
        return A_init

class SCGAT(nn.Module):
    def __init__(self, in_channels, out_channels, stage_dim=64, num_stages=5, num_heads=4):
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
        
        self.att_scorer = nn.Linear(2 * self.d_k + stage_dim, 1, bias=False)
        self.lambda_prior = nn.Parameter(torch.ones(num_heads))

    def forward(self, x, A_prior):
        B, C, _ = x.shape
        x_proj = self.lin_project(x).view(B, C, self.num_heads, self.d_k)
        
        s_logits = self.stage_mlp(x.mean(dim=1))
        s_prob = F.softmax(s_logits, dim=-1)
        c = self.mlp_c(s_prob) 
        
        x_dst = x_proj.unsqueeze(2).expand(-1, -1, C, -1, -1) 
        x_src = x_proj.unsqueeze(1).expand(-1, C, -1, -1, -1) 
        c_rep = c.view(B, 1, 1, 1, -1).expand(-1, C, C, self.num_heads, -1)
        
        att_input = torch.cat([x_src, x_dst, c_rep], dim=-1) 
        e = F.leaky_relu(self.att_scorer(att_input).squeeze(-1), 0.2) 
        
        prior_term = self.lambda_prior.view(1, 1, 1, -1) * torch.log(A_prior.unsqueeze(0).unsqueeze(-1) + 1e-9)
        combined_e = e + prior_term
        
        alpha = F.softmax(combined_e, dim=2) 
        
        alpha = alpha.unsqueeze(-1) 
        x_val = x_proj.unsqueeze(1).expand(-1, C, -1, -1, -1) 
        out = (alpha * x_val).sum(dim=2) 
        
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
        
        tau = self.node_time_pos.unsqueeze(1) - self.node_time_pos.unsqueeze(0) 
        t_enc = self.time_mlp(tau) 
        
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