import torch

def sinkhorn_knopp(C, epsilon=0.01, max_iter=100):
    """
    GPU 并行的 Sinkhorn-Knopp 算法。
    计算最优传输的熵正则化耦合矩阵 P。
    时间复杂度 O(N^2 * max_iter)，比匈牙利算法 O(N^3) 快得多，支持极大 Batch Size。
    
    参数:
        C: [B, B] 代价矩阵 (距离矩阵)
        epsilon: 熵正则化系数 (越小越逼近精确 OT，但容易数值不稳定)
        max_iter: 最大迭代次数
    """
    B = C.shape[0]
    device = C.device
    
    # 核心：计算 Gibbs Kernel K
    # 为了数值稳定，减去行最小值
    C_min = torch.min(C, dim=1, keepdim=True)[0]
    K = torch.exp(-(C - C_min) / epsilon)
    
    # 初始化边际分布 (由于我们是均匀匹配，每个样本权重均为 1/B)
    a = torch.ones(B, device=device) / B
    b = torch.ones(B, device=device) / B
    
    # 初始化 scaling vectors
    u = torch.ones(B, device=device) / B
    v = torch.ones(B, device=device) / B
    
    # Sinkhorn 迭代
    for _ in range(max_iter):
        u = a / (torch.matmul(K, v) + 1e-8)
        v = b / (torch.matmul(K.t(), u) + 1e-8)
        
    # 重构最终的最优传输矩阵 P
    P = u.unsqueeze(1) * K * v.unsqueeze(0)
    return P

def compute_ot_coupling(x0, x1, epsilon=0.01):
    """
    基于 GPU Sinkhorn 的最优传输耦合。
    """
    B = x0.shape[0]
    # 展平计算距离矩阵
    x0_flat = x0.view(B, -1)
    x1_flat = x1.view(B, -1)
    
    # 完全在 GPU 上计算代价矩阵 [B, B]
    C = torch.cdist(x0_flat, x1_flat, p=2)
    
    # 使用 Sinkhorn 算法计算传输概率矩阵 P
    P = sinkhorn_knopp(C, epsilon=epsilon)
    
    # 对于每个高斯噪声，根据 P 矩阵的概率分布找到最匹配的目标图像
    # 取概率最大项作为确定性耦合 (也可以用 torch.multinomial 随机采样)
    col_ind = torch.argmax(P, dim=1)
    
    # 重新排列 x0 以对齐 x1
    x0_coupled = torch.empty_like(x0)
    x0_coupled[col_ind] = x0
    return x0_coupled

def flow_matching_loss(model, x1, y=None, use_ot=True):
    """
    OT-Flow Matching 损失函数 (支持带标签的条件生成 CFG)。
    计算目标向量场 u_t 与模型预测向量场 v_pred 之间的 MSE 损失。
    """
    B = x1.shape[0]
    device = x1.device
    
    x0 = torch.randn_like(x1)
    
    # [工业级升级] 全程 GPU 加速的 OT 耦合
    if use_ot:
        x0 = compute_ot_coupling(x0, x1)
        
    t = torch.rand((B, 1, 1, 1), device=device)
    
    x_t = (1 - t) * x0 + t * x1
    u_t = x1 - x0
    
    t_model = t.squeeze()
    
    # 传入条件标签 y
    v_pred = model(x_t, t_model, y)
    
    loss = torch.nn.functional.mse_loss(v_pred, u_t)
    return loss
