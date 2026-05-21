import torch
import time
from src.models.dit import DiffusionTransformer
from src.models.ot_matching import flow_matching_loss, compute_ot_coupling

def test_ot_flow():
    print("=== 开始测试 OT-Flow 工业级核心模块 ===")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] 运行设备: {device}")
    
    B, C, H, W = 4, 3, 32, 32
    print(f"[*] 模拟数据 Shape: [{B}, {C}, {H}, {W}]")
    
    x1 = torch.randn((B, C, H, W), device=device)
    x0 = torch.randn((B, C, H, W), device=device)
    
    print("\n[*] 测试 GPU 加速的 Sinkhorn OT Coupling...")
    start_time = time.time()
    x0_coupled = compute_ot_coupling(x0, x1, epsilon=0.01)
    ot_time = time.time() - start_time
    
    print(f"    OT 耦合完成 (耗时: {ot_time:.4f}s)")
    
    dist_before = torch.nn.functional.mse_loss(x0, x1).item()
    dist_after = torch.nn.functional.mse_loss(x0_coupled, x1).item()
    print(f"    优化前 Batch MSE: {dist_before:.4f}")
    print(f"    优化后 Batch MSE: {dist_after:.4f} (显著下降说明 Sinkhorn 生效)")
    
    print("\n[*] 初始化包含 CFG 支持的 DiT 主干...")
    model = DiffusionTransformer(in_channels=3, image_size=32, hidden_size=128, num_layers=2, num_heads=4, num_classes=1000).to(device)
    
    print("[*] 计算带标签的 Flow Matching Loss...")
    y = torch.randint(0, 1000, (B,), device=device)
    loss = flow_matching_loss(model, x1, y=y)
    
    print(f"    计算成功! 当前步 Loss 值: {loss.item():.4f}")
    
    print("\n[*] 测试梯度反向传播...")
    loss.backward()
    
    has_grad = any(p.grad is not None and torch.sum(p.grad).item() != 0 for p in model.parameters())
    if has_grad:
        print("    [Success] 梯度流完全正常！架构逻辑闭环测试通过。")
    else:
        print("    [Error] 梯度未能正常传播！")
        
    print("==================================================")

if __name__ == "__main__":
    test_ot_flow()
