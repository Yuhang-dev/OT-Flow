import torch
import scipy.optimize

def compute_ot_coupling(x0, x1):
    """
    计算基于最优传输 (Optimal Transport) 的确定性耦合。
    """
    B = x0.shape[0]
    x0_flat = x0.view(B, -1)
    x1_flat = x1.view(B, -1)
    
    # 将计算放在 CPU 上进行，因为匈牙利算法不支持 GPU
    cost_matrix = torch.cdist(x0_flat, x1_flat, p=2).cpu().detach().numpy()
    row_ind, col_ind = scipy.optimize.linear_sum_assignment(cost_matrix)
    
    x0_coupled = torch.empty_like(x0)
    x0_coupled[col_ind] = x0[row_ind]
    return x0_coupled

def flow_matching_loss(model, x1, use_ot=True, condition=None):
    """
    OT-Flow Matching 损失函数。
    计算目标向量场 u_t 与模型预测向量场 v_pred 之间的 MSE 损失。
    """
    B = x1.shape[0]
    device = x1.device
    
    x0 = torch.randn_like(x1)
    
    if use_ot:
        x0 = compute_ot_coupling(x0, x1)
        
    t = torch.rand((B, 1, 1, 1), device=device)
    
    x_t = (1 - t) * x0 + t * x1
    u_t = x1 - x0
    
    t_model = t.squeeze()
    v_pred = model(x_t, t_model, condition)
    
    loss = torch.nn.functional.mse_loss(v_pred, u_t)
    return loss
