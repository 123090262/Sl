import torch
import torch.nn as nn
import torch.nn.functional as F

class SleepGATNet(nn.Module):
    def __init__(self, channel_names, fs=200, num_stages=5, feature_dim=64, hid_dim=64):
        super(SleepGATNet, self).__init__()
        self.C = len(channel_names)
        self.fs = fs
        
        # 1. 结构先验构建器 (PLV + Geodesic)
        self.prior_builder = PriorMatrixBuilder(channel_names=channel_names)
        
        # 2. 浅层 TCN 特征提取器 (原始波形编码)
        self.feature_extractor = FeatureExtractor(
            num_channels=self.C, 
            fs=self.fs,
            feature_dim=feature_dim
        )
        
        # 3. 双路图注意力网络
        self.tagat = TAGAT(
            in_dim=feature_dim, 
            hid_dim=hid_dim, 
            out_dim=hid_dim, 
            num_nodes=self.C,
            channel_names=channel_names
        )
        
        self.scgat = SCGAT(
            in_channels=feature_dim, 
            out_channels=hid_dim, 
            stage_dim=32, 
            num_stages=num_stages
        )
        
        # 4. 门控融合
        self.gate_fusion = GatedFusion(dim=hid_dim)
        
        # 5. 通道序列建模 (BiLSTM)
        self.bilstm = BiLSTM(input_dim=hid_dim, hidden_dim=hid_dim)
        
        # 6. 通道注意力
        self.channel_att = ChannelAttention(dim=hid_dim * 2)
        
        # 7. 分类头
        self.classifier = nn.Sequential(
            nn.Linear(hid_dim * 2, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_stages)
        )

    def forward(self, x_raw):
        """
        Input x_raw: (B, C, 6000)
        """
        # Step 1: 构建先验邻接矩阵 A_init
        A_init = self.prior_builder(x_raw) # (C, C)
        
        # Step 2: 提取局部时空特征
        # x_feat: (B, C, 30, feature_dim)
        x_feat = self.feature_extractor(x_raw)
        
        # Step 3: 并行执行 TAGAT 与 SCGAT
        # h_tagat: (B, C, hid_dim)
        h_tagat = self.tagat(x_feat, A_init)
        # h_scgat: (B, C, hid_dim), s_prob: (B, num_stages)
        h_scgat, s_prob = self.scgat(x_feat, A_init)
        
        # Step 4: 门控融合
        h_fused = self.gate_fusion(h_tagat, h_scgat) # (B, C, hid_dim)
        
        # Step 5: BiLSTM 建模通道间依赖
        h_gru = self.bilstm(h_fused) # (B, C, hid_dim * 2)
        
        # Step 6: 通道注意力增强
        h_att, att_weights = self.channel_att(h_gru) # (B, C, hid_dim * 2)
        
        # Step 7: 全局平均池化 (GAP) 与 分类
        h_gap = torch.mean(h_att, dim=1) # (B, hid_dim * 2)
        logits = self.classifier(h_gap) # (B, num_stages)
        
        return logits, s_prob, A_init, att_weights


class ShallowTCN(nn.Module):
    def __init__(self, fs, feature_dim=64):
        """
        浅层 TCN 单元：用于将 1秒(200点or100点) 的原始 EEG 信号编码为特征向量
        """
        super(ShallowTCN, self).__init__()
        # 根据采样率动态计算 kernel_size (目标是覆盖约 0.32s 的信号)
        k1 = int(64 * (fs / 200))# 200Hz->64, 100Hz->32
        s1 = int(6 * (fs / 200)) # 200Hz->6, 100Hz->3
        k2 = 8 if fs == 200 else 4 # 第二层卷积核适配
        
        # Block 1: 提取低频和大幅度波形 (如 Delta 波, K-complex)---粗粒度
        # kernel_size 较大，stride 较大以降低维度（降采样）
        self.block1 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=32, kernel_size=k1, stride=s1, padding=k1//2),
            nn.BatchNorm1d(32), #32个特征
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        # 输入 (N, 1, 200) -> Conv -> (N, 32, ~33)
        
        # Block 2: 提取高频细节 (如 Spindles, Alpha 波)
        self.block2 = nn.Sequential(
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=k2, stride=1, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Dropout(0.5)
        )
        # 输入 (N, 32, 33) -> Conv -> (N, 64, 33) -> Pool -> (N, 64, 16)
        
        #200点大约变成了长度 16 (取决于padding细节)
        self._to_linear = self._get_linear_size(fs)
        
        # 全连接层映射到目标特征维度
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self._to_linear, feature_dim),
            nn.ReLU() # 保持非线性，适合后续接 GNN
        )
    def _get_linear_size(self, fs):
        """通过模拟转发计算 Block2 输出的维度尺寸"""
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, fs) # 模拟 1秒 数据输入
            output = self.block1(dummy_input)
            output = self.block2(output)
            return output.numel() # 返回展平后的总元素个数
        
    def forward(self, x):
        # x shape: (N, 1, 200)
        x = self.block1(x)
        x = self.block2(x)
        out = self.fc(x)
        return out


class FeatureExtractor(nn.Module):
    def __init__(self, num_channels=6, fs = 200, time_steps=30, feature_dim=64):
        super(FeatureExtractor, self).__init__()
        self.C = num_channels
        self.fs = fs
        self.T_prime = time_steps   # 30
        
        # 初始化 TCN 编码器
        self.tcn = ShallowTCN(fs = self.fs, feature_dim=feature_dim)

    def forward(self, x):
        """
        输入 x: (B, C, 6000)
        输出 out: (B, C, T', feature_dim) 
        这样每个时间步 T' 都有一个 (C, feature_dim) 的图节点特征矩阵
        """
        B, C, T_total = x.size()
        
        # 1. 滑窗 / Reshape to (B, C, 30, 200)
        if T_total != self.T_prime * self.fs:
            raise ValueError(f"输入数据长度 {T_total} 与设定 {self.T_prime}*{self.d} 不匹配")
            
        x_reshaped = x.view(B, C, self.T_prime, self.fs)
        
        # 2. 维度合并以进行并行 TCN 提取 (B, C, 30, 200) -> (B * C * 30, 1, 200)
        # TCN 需要输入 (N, 1, Length)
        # 把 B, C, T' 全部合并到 N 维度，让网络对"每个通道的每秒数据"独立提取特征
        x_batch_in = x_reshaped.view(B * C * self.T_prime, 1, self.fs)
        
        # 3. 通过 TCN 提取特征---(B * C * 30, feature_dim)
        features = self.tcn(x_batch_in)
        
        # 4. 还原维度 --- (B, C, T', feature_dim)
        out = features.view(B, C, self.T_prime, -1)
        
        return out
    

class PriorMatrixBuilder(nn.Module):
    def __init__(self, channel_names, sigma=1.0, beta=10.0, tau=0.5):
        super(PriorMatrixBuilder, self).__init__()
        self.C = len(channel_names)
        self.sigma = sigma
        self.beta = beta
        self.tau = tau
        
        # 1. 10-20 系统标准坐标 
        coords = {
            "C3-A2": [-0.707, 0.0, 0.707],  "C4-A1": [0.707, 0.0, 0.707],
            "O1-A2": [-0.4, -0.9, 0.1],     "O2-A1": [0.4, -0.9, 0.1],
            "F3-A2": [-0.4, 0.8, 0.4],      "F4-A1": [0.4, 0.8, 0.4],
            "EEG Fpz-Cz":[0.0, 0.95, 0.2],  "EEG Pz-Oz":[0.0,-0.95,0.2]  
                }
        pos = torch.tensor([coords[name] for name in channel_names], dtype=torch.float32) #(C,3)张量
        # 预计算几何先验 Ageo
        dist = torch.cdist(pos, pos, p=2) #计算 x1 中每个点与 x2 中每个点的「p- 范数距离」，最终返回一个「点对距离矩阵」
            #p=2，L2欧式距离；计算 pos 中所有点与自身所有点的两两距离（自距离矩阵）
        self.register_buffer('A_geo', torch.exp(-(dist**2) / (self.sigma**2))) 
        #通过高斯核（RBF 核）将「距离」转化为「几何相似度矩阵」 !距离越近值越接近1! 并将该矩阵注册为模型的非参数缓冲区
        
        # 可学习参数
        self.weight_fusion = nn.Parameter(torch.tensor([0.5, 0.5])) # 初始各占一半,动态调整几何与功能的相加关系
        self.E = nn.Parameter(torch.randn(self.C, self.C) * 0.01) #微小的随机噪声初始化，用于微调融合后的邻接矩阵

    def compute_plv(self, x_raw):
        """
        计算生理 PLV 矩阵
        输入 x_raw: (B, C, 6000) 预处理后的原始信号
        """
        B, C, T = x_raw.shape
        
        # --- Hilbert 变换提取相位 ---
        Xf = torch.fft.fft(x_raw, dim=-1)
        h = torch.zeros(T, device=x_raw.device) #频域滤波器
        if T % 2 == 0: #6000
            h[0] = h[T//2] = 1 #DC 和 Nyquist，保留
            h[1:T//2] = 2 #正频率翻倍
        else:
            h[0] = 1; h[1:(T+1)//2] = 2
        
        analytic_signal = torch.fft.ifft(Xf * h, dim=-1)
        phase = torch.angle(analytic_signal) # 得到每个通道在每个时间点的瞬时相位phi(t) (B,C,T)
        
        # --- 计算 PLV ---
        # 扩展维度进行两两对比 (B, C, C, T)
        phase_diff = phase.unsqueeze(2) - phase.unsqueeze(1) #(B, C, 1, T) - (B, 1, C, T) → (B, C, C, T)
        
        # 利用欧拉公式
        real_avg = torch.cos(phase_diff).mean(dim=-1)
        imag_avg = torch.sin(phase_diff).mean(dim=-1) #(B,C,C)
        # 求和均值
        plv_matrix = torch.sqrt(real_avg**2 + imag_avg**2) # 取模长得到 PLV: sqrt(real^2 + imag^2)
        return plv_matrix # (B, C, C)

    def forward(self, x_raw):
        """
        输入 x_raw: (B, C, 6000)
        输出 A_init: (C, C)
        """
        # 1. 计算 Batch 内平均的 PLV 矩阵
        A_plv = self.compute_plv(x_raw).mean(dim=0) # (C, C)
        
        # 2. 权重融合 (几何 + 功能)
        w = torch.softmax(self.weight_fusion, dim=0)
        A_0 = w[0] * self.A_geo + w[1] * A_plv
        
        # 3. 引入可学习微调矩阵 E （可训练参数）
        A_init = A_0 * F.softplus(self.E) #激活函数（ln(1+e^x))是为了保证调节系数始终为正数
        
        # 4. 规范化处理
        A_init = (A_init + A_init.T) / 2 # 对称化,强制消除方向性的差异
        A_init.fill_diagonal_(0)         # 对角线置零，防止自身特征主导
        
        # 5. Soft-threshold 稀疏化
        # 使用 sigmoid(beta * (A - tau)) 实现可微阈值筛选
        A_init = torch.sigmoid(self.beta * (A_init - self.tau))
        
        return A_init

class SCGAT(nn.Module):
    def __init__(self, in_channels, out_channels, stage_dim, num_stages,lambda_prior=0.5):
        super(SCGAT, self).__init__()
        self.in_channels = in_channels   #输入特征维度D
        self.out_channels = out_channels #输出.......d'
        self.lambda_prior = nn.Parameter(torch.tensor([lambda_prior])) #先验矩阵元素的权重系数
        
        # 1. 节点特征线性映射 
        self.lin_project = nn.Linear(in_channels, out_channels)  #W h -> Z
        
        # 2. Stage Encoder: 从节点特征池化后预测阶段 
        self.stage_mlp = nn.Sequential(
            nn.Linear(in_channels, 64), #降维
            nn.ReLU(),
            nn.Linear(64, num_stages), #5
            nn.Softmax(dim=-1)  #转化为概率分布
        )
        
        # 3. 将阶段条件概率映射为调制向量 c 
        self.mlp_c = nn.Sequential(
            nn.Linear(num_stages, stage_dim),
            nn.ReLU(),
            nn.Linear(stage_dim, stage_dim)
        )
        
        # 4. 注意力计算参数
         
        # 拼接 [Zi || Zj || c]，所以输入维度是 2 * out_channels + stage_dim
        self.att_weight = nn.Parameter(torch.Tensor(1, 1, 2 * out_channels + stage_dim))
        #Parameter:可学习参数的类；torch.Tensor未初始化的张量，形状为（1，1，2*out......）
        
        self.leaky_relu = nn.LeakyReLU(0.2)
        
        self.reset_parameters()

    def reset_parameters(self): #对可学习参数进行统一初始化 （Xavier 均匀初始化）
        nn.init.xavier_uniform_(self.lin_project.weight) #W（in，out）
        nn.init.xavier_uniform_(self.att_weight) #a （1，1，C）

    def forward(self, x, A_init):
        """
        输入:
        x: (B, C, T', d) - 原始节点特征矩阵
        A_init: (C, C) - 初始先验邻接矩阵
        """
        B, C, T, D = x.size()
        
        # --- 维度降维 (Channel-Temporal Pooling) ---
        # 将 (B, C, T', d) 降维成 (B, C, d), 这里采用 Mean Pooling
        x_reduced = torch.mean(x, dim=2) # (B, C, D)
        
        # --- Step 1: 线性映射 ---
        Z = self.lin_project(x_reduced) # (B, C, d')  Zi=W*hi，只对最后一个维度线性变换
                
        # --- Stage Encoder 预分类器 ---
        x_global = torch.mean(x_reduced, dim=1) # (B,C,D) -> (B, D)
        s = self.stage_mlp(x_global) # (B, num_stages) 阶段概率分布向量s
              
        # --- Step 2: 调制向量 c ---
        c = self.mlp_c(s) # (B, stage_dim)
        
        # --- Step 3: 计算注意力 Logits (e_ij) ---
        # 1. 准备 Zi: 每一行都是相同的通道特征
        Z_i = Z.unsqueeze(2).expand(-1, -1, C, -1) #unsqueeze(2)在第2维插入大小为1的维度；-1保持该维不变

        # 2. 准备 Zj: 每一列都是相同的通道特征
        Z_j = Z.unsqueeze(1).expand(-1, C, -1, -1)  #（B,6,6,d'）Zi与Zj形状一致；expand拓展维度，不复制数据

        # 3. 准备 c: 将阶段调制向量广播到所有节点对 (i, j)
        # (B,stage_dim) -> (B, 1, 1,stage_dim)-> (B, 6, 6,stage_dim)
        c_ext = c.unsqueeze(1).unsqueeze(2).expand(-1, C, C, -1)

        # 4. 拼接 [Zi || Zj || c]
        # 结果形状: (B, 6, 6, 2*d' + dc)
        features = torch.cat([Z_i, Z_j, c_ext], dim=-1) #沿着最后一维拼接 -> (B,6,6,2d'+stage_dim)

        # 5. 计算 logits e_ij
        e = self.leaky_relu((features*self.att_weight).sum(dim=-1)) #点积：逐元素相乘再在最后一维求和
        #e: (B, 6, 6)
        
        # --- Step 4: 结合初始邻接先验 ---
        eps = 1e-8
        A_prior_log = torch.log(A_init + eps)
        e_tilde = e + self.lambda_prior * A_prior_log.unsqueeze(0) # (1, C, C)，广播机制自动将1复制成B -> (B,C,C)
        
        # --- Step 5: 归一化得到动态权重 alpha ---
        alpha = F.softmax(e_tilde, dim=-1) # (B, C, C),沿着j归一化，保证对于每一个目标节点i，所有邻居j权重和为1
        
        # --- Step 6: 聚合得到新特征 h' ---
        h_SCGAT = torch.matmul(alpha, Z) # (B, C, d')
        
        return h_SCGAT, s # 返回新特征和阶段预测结果
    
class TAGAT(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, num_nodes, channel_names=None, time_steps=30, t_enc_dim=16, alpha=0.2):
        """
        :param in_dim: 输入特征维度 d
        :param hid_dim: 内部投影维度 d'
        :param out_dim: 最终输出维度 D (对齐 SCGAT)
        :param num_nodes: 节点数 C=6
        :param time_steps: 时间片数量 T'
        :param t_enc_dim: 时间编码维度 d_tau
        """
        super(TAGAT, self).__init__()
        self.C = num_nodes
        self.T = time_steps
        self.d_prime = hid_dim
        self.D = out_dim

        # 1. 节点特征线性投影 Wh
        self.W_h = nn.Linear(in_dim, hid_dim)

        # 2. 可学习的时间位置参数 (用于计算 Δt_ij)
        if channel_names:# 根据生理学位置设定初始值
            init_pos = torch.zeros(num_nodes, 1)  #(C,1)
            for i, name in enumerate(channel_names):
                name = name.upper()
                if 'F3' or 'F4' in name:   # 额叶：初始化为较早的时间点
                    init_pos[i] = 1.0
                elif 'C' in name: # 中央区
                    init_pos[i] = 0.5
                elif 'O1' or 'O2' or 'PZ' in name: # 枕叶：初始化为较晚的时间点
                    init_pos[i] = 0.0
            self.node_time_pos = nn.Parameter(init_pos)
        else:
            self.node_time_pos = nn.Parameter(torch.randn(num_nodes, 1))
        
        # 3. 时间编码函数 f_time (MLP)
        self.f_time = nn.Sequential(
            nn.Linear(1, t_enc_dim),
            nn.ReLU(),
            nn.Linear(t_enc_dim, t_enc_dim)
        )

        # 4. 注意力向量 a (2*d' + d_tau)
        self.a = nn.Parameter(torch.Tensor(2 * hid_dim + t_enc_dim, 1))
        
        # 5. A_init 的调节权重 lambda
        self.lambd = nn.Parameter(torch.ones(1)) #形状为（1，）初始值为1.0的一维张量
        
        # 6. 最终维度投影 (T'*d' -> D)
        self.output_proj = nn.Linear(time_steps * hid_dim, out_dim)

        self.leakyrelu = nn.LeakyReLU(alpha) #0.2
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W_h.weight)
        #nn.init.xavier_uniform_(self.node_time_pos)
        nn.init.xavier_uniform_(self.a)   

    def forward(self, x, A_init):
        """
        :param x: 输入张量 (B, C, T', d)
        :param A_init: 初始邻接矩阵 (C, C)
        :return: (B, C, D)
        """
        B, C, T, d = x.size()
        
        # Step 1: 节点特征投影 Z = Wh * H (B, C, T, d')
        z = self.W_h(x) 

        # Step 2: 计算相对时间编码 tau_ij
        # 计算所有节点两两之间的时间差 Δt (C, C, 1)
        rel_time = self.node_time_pos.unsqueeze(1) - self.node_time_pos.unsqueeze(0) 
        tau = self.f_time(rel_time) # (C, C, d_tau)

        # Step 3 & 4: 注意力 Logits 计算广播 z 和 tau
        # z_i: (B, T, C, 1, d'), z_j: (B, T, 1, C, d')
        z_expanded = z.transpose(1, 2) # (B, T, C, d')
        z_i = z_expanded.unsqueeze(3).expand(-1, -1, -1, C, -1)
        z_j = z_expanded.unsqueeze(2).expand(-1, -1, C, -1, -1)
        tau_expanded = tau.unsqueeze(0).unsqueeze(0).expand(B, T, -1, -1, -1)
        
        # tau 扩展为 (B, T, C, C, d_tau)
        tau_expanded = tau.unsqueeze(0).unsqueeze(0).repeat(B, T, 1, 1, 1)

        # 拼接 [Z_i || Z_j || tau] -> (B, T, C, C, 2*d' + d_tau)
        combined = torch.cat([z_i, z_j, tau_expanded], dim=-1)
        
        # e_ij = LeakyReLU(a^T * combined)
        e = self.leakyrelu(torch.matmul(combined, self.a).squeeze(-1)) # (B, T, C, C)

        # Step 5: 融入 A_init 先验
        # e_tilde = e + lambda * log(A_init + eps)
        eps = 1e-8
        e_tilde = e + self.lambd * torch.log(A_init + eps).unsqueeze(0).unsqueeze(0) #（1，1，C,C）-> (B,T,C,C)广播机制自动

        # Step 6: Softmax 得到注意力权重 alpha
        alpha = F.softmax(e_tilde, dim=-1) # (B, T, C, C)

        # Step 7: 节点特征更新 H' = alpha * Z
        # (B, T, C, C) * (B, T, C, d') -> (B, T, C, d')
        h_prime = torch.matmul(alpha, z_expanded) 
        h_prime = h_prime.transpose(1, 2) # (B, C, T, d')

        # Step 8: 汇聚所有时间步并对齐维度 (B, C, T*d') -> (B, C, D)
        h_flat = h_prime.reshape(B, C, -1)
        out = self.output_proj(h_flat)

        return out
    
class GatedFusion(nn.Module):
    def __init__(self, dim):
        super(GatedFusion, self).__init__()
        self.fc = nn.Linear(dim * 2, dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x1, x2): # x1, x2: (B, C, D)
        combined = torch.cat([x1, x2], dim=-1) #(B,C,2D)
        z = self.sigmoid(self.fc(combined)) #(B,C,D)
        return z * x1 + (1 - z) * x2

class BiLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=1, dropout=0.2):
        """
        双向 LSTM (BiLSTM) 模块
        :param input_dim: 输入特征的维度 (例如 GAT 输出的 hid_dim)
        :param hidden_dim: LSTM 内部隐藏层的维度 (单向)
        :param num_layers: LSTM 的层数
        :param dropout: 层间 Dropout 概率
        """
        super(BiLSTM, self).__init__()
        
        # 定义双向 LSTM
        # batch_first=True 表示输入形状为 (Batch, Seq_len, Feature)
        # bidirectional=True 开启双向
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # 由于是双向，输出维度会翻倍
        self.out_dim = hidden_dim * 2

    def forward(self, x):
        """
        :param x: 输入张量，形状为 (Batch, Seq_len, Input_dim)
                  在你的模型中，Seq_len 对应通道数 C
        :return: 融合了双向信息的序列特征，形状为 (Batch, Seq_len, Hidden_dim * 2)
        """
        # output 包含序列中每个时间步的最后隐状态：(Batch, Seq_len, Hidden_dim * 2)
        # h_n 包含最后时刻的隐状态：(num_layers * 2, Batch, Hidden_dim)
        # c_n 包含最后时刻的细胞状态：(num_layers * 2, Batch, Hidden_dim)
        output, (h_n, c_n) = self.lstm(x)
        
        return output
    
class ChannelAttention(nn.Module):
    def __init__(self, dim):
        super(ChannelAttention, self).__init__()
        self.att = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.ReLU(),
            nn.Linear(dim // 2, 1),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        # x: (B, C, D)
        weights = self.att(x) # (B, C, 1)每个通道的权重
        return x * weights, weights