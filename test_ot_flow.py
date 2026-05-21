import torch
import time
from ot_flow import MiniDiT, flow_matching_loss, compute_ot_coupling

def test_ot_flow():
    print("=== 开始测试 OT-Flow (最优传输流匹配) 核心模块 ===")
    
    # 1. 模拟设备与数据
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] 运行设备: {device}")
    
    B, C, H, W = 4, 3, 32, 32  # 模拟一个小 Batch 的图像数据
    print(f"[*] 模拟数据 Shape: [{B}, {C}, {H}, {W}]")
    
    # 模拟真实数据 x1 (比如通过 VAE 编码后的 Latent)
    x1 = torch.randn((B, C, H, W), device=device)
    # 模拟随机噪声 x0
    x0 = torch.randn((B, C, H, W), device=device)
    
    # 2. 测试 OT Coupling (匈牙利算法重排)
    start_time = time.time()
    x0_coupled = compute_ot_coupling(x0, x1)
    ot_time = time.time() - start_time
    
    print(f"\n[*] OT 耦合测试完成 (耗时: {ot_time:.4f}s)")
    print(f"    原始 x0 shape: {x0.shape}")
    print(f"    耦合后 x0 shape: {x0_coupled.shape}")
    
    # 计算重排前后的总距离对比 (理论上重排后距离应变小)
    dist_before = torch.nn.functional.mse_loss(x0, x1).item()
    dist_after = torch.nn.functional.mse_loss(x0_coupled, x1).item()
    print(f"    优化前 Batch MSE: {dist_before:.4f}")
    print(f"    优化后 Batch MSE: {dist_after:.4f} (显著下降说明 OT 生效)")
    
    # 3. 测试模型架构与 Loss 计算
    print("\n[*] 初始化 Mini-DiT (Diffusion Transformer) 主干...")
    model = MiniDiT(in_channels=3, hidden_size=128, num_layers=2, num_heads=4).to(device)
    
    print("[*] 计算 Flow Matching Loss (包含随机采样 t 与插值)...")
    loss = flow_matching_loss(model, x1)
    
    print(f"    计算成功! 当前步 Loss 值: {loss.item():.4f}")
    
    # 4. 测试梯度反向传播
    print("\n[*] 测试梯度反向传播...")
    loss.backward()
    
    # 检查梯度是否正常流回模型
    has_grad = any(p.grad is not None and torch.sum(p.grad).item() != 0 for p in model.parameters())
    if has_grad:
        print("    [Success] 梯度流完全正常！OT-Flow 架构逻辑闭环测试通过。")
    else:
        print("    [Error] 梯度未能正常传播！")
        
    print("==================================================")

if __name__ == "__main__":
    test_ot_flow()
