import torch
import torch.nn as nn
import math
import scipy.optimize

def compute_ot_coupling(x0, x1):
    """
    计算基于最优传输 (Optimal Transport) 的确定性耦合。
    将噪声 x0 重新排列，使其在批次内与目标数据 x1 的总传输成本 (MSE) 最小。
    这对 Flow Matching (流匹配) 至关重要，能够拉直轨迹，加速训练收敛。
    
    参数:
        x0: [B, C, H, W] 基础分布 (高斯噪声)
        x1: [B, C, H, W] 目标分布 (真实数据)
    返回:
        x0_coupled: 重新排列后的 x0
    """
    B = x0.shape[0]
    # 将空间维度展平计算距离矩阵
    x0_flat = x0.view(B, -1)
    x1_flat = x1.view(B, -1)
    
    # 计算代价矩阵 (L2 距离)
    # x0_flat: [B, D], x1_flat: [B, D] -> cost_matrix: [B, B]
    cost_matrix = torch.cdist(x0_flat, x1_flat, p=2).cpu().detach().numpy()
    
    # 匈牙利算法求解二分图最小权匹配 (Linear Sum Assignment)
    row_ind, col_ind = scipy.optimize.linear_sum_assignment(cost_matrix)
    
    # 根据求解出的排列重新映射 x0
    # 由于 row_ind 总是 0..B-1, 我们只需按 col_ind 重排 x1 或按 reverse_col_ind 重排 x0
    # 这里我们重排 x0 去对齐 x1
    x0_coupled = torch.empty_like(x0)
    x0_coupled[col_ind] = x0[row_ind]
    return x0_coupled

def flow_matching_loss(model, x1, condition=None):
    """
    OT-Flow Matching 损失函数。
    计算目标向量场 u_t 与模型预测向量场 v_pred 之间的 MSE 损失。
    """
    B = x1.shape[0]
    device = x1.device
    
    # 1. 采样高斯噪声
    x0 = torch.randn_like(x1)
    
    # 2. 最优传输耦合 (OT Coupling)
    # 将随机耦合转为最优确定性耦合，大幅降低方差
    x0 = compute_ot_coupling(x0, x1)
    
    # 3. 随机采样时间步 t ~ U[0, 1]
    t = torch.rand((B, 1, 1, 1), device=device)
    
    # 4. 在直线路径上插值计算 x_t
    x_t = (1 - t) * x0 + t * x1
    
    # 5. 计算真实的向量场 (Velocity Field)
    u_t = x1 - x0
    
    # 6. 模型预测当前时间步的向量场
    # 将 t 调整为 [B] 传给模型
    t_model = t.squeeze()
    v_pred = model(x_t, t_model, condition)
    
    # 7. 计算回归损失 (MSE Loss)
    loss = torch.nn.functional.mse_loss(v_pred, u_t)
    return loss

def timestep_embedding(timesteps, dim, max_period=10000):
    """标准的正弦/余弦时间步嵌入 (Sinusoidal Timestep Embedding)"""
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding

class MiniDiT(nn.Module):
    """
    一个极简的 Diffusion Transformer (DiT) 主干网络。
    用于预测 Flow Matching 中的向量场。
    """
    def __init__(self, in_channels=3, hidden_size=256, num_layers=4, num_heads=4):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_size = hidden_size
        
        # 图像特征映射
        self.proj = nn.Conv2d(in_channels, hidden_size, kernel_size=1)
        
        # 时间步嵌入 MLP
        self.time_embed = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size)
        )
        
        # Transformer 编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size, 
            nhead=num_heads, 
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 输出向量场映射
        self.out_proj = nn.Conv2d(hidden_size, in_channels, kernel_size=1)
        
    def forward(self, x, t, condition=None):
        """
        x: [B, C, H, W]
        t: [B]
        """
        B, C, H, W = x.shape
        
        # 1. 图像投影 [B, D, H, W]
        x_emb = self.proj(x) 
        
        # 2. 时间嵌入 [B, D, 1, 1]
        t_emb = timestep_embedding(t, self.hidden_size)
        t_emb = self.time_embed(t_emb).unsqueeze(-1).unsqueeze(-1)
        
        # 3. 融合时间特征
        h = x_emb + t_emb
        
        # 4. 展平为序列进行 Transformer 交互
        h_flat = h.flatten(2).transpose(1, 2) # [B, H*W, D]
        h_out = self.transformer(h_flat)
        h_out = h_out.transpose(1, 2).view(B, self.hidden_size, H, W)
        
        # 5. 投影到原通道输出向量场
        out = self.out_proj(h_out)
        return out
