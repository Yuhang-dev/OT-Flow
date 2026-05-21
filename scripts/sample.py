import argparse
import torch
import torchvision
from src.models.dit import DiffusionTransformer

@torch.no_grad()
def get_velocity_cfg(model, x, t, y, w_cfg=4.0, num_classes=1000):
    """
    [工业级升级] Classifier-Free Guidance (CFG) 向量场推断
    v_cfg = v_uncond + w * (v_cond - v_uncond)
    """
    if w_cfg == 1.0 or y is None:
        return model(x, t, y)
        
    B = x.shape[0]
    device = x.device
    
    # 构造无条件标签 (Unconditional Token = num_classes)
    y_uncond = torch.full((B,), num_classes, device=device, dtype=torch.long)
    
    # 拼接批量推断以优化性能
    x_double = torch.cat([x, x], dim=0)
    t_double = torch.cat([t, t], dim=0)
    y_double = torch.cat([y, y_uncond], dim=0)
    
    v_pred = model(x_double, t_double, y_double)
    v_cond, v_uncond = v_pred.chunk(2, dim=0)
    
    return v_uncond + w_cfg * (v_cond - v_uncond)


@torch.no_grad()
def sample_euler(model, shape, device, y=None, steps=10, w_cfg=4.0):
    """一阶 Euler 积分器 (速度快，适用于大步数)"""
    x = torch.randn(shape, device=device)
    t_steps = torch.linspace(0, 1, steps + 1, device=device)
    dt = 1.0 / steps
    
    for i in range(steps):
        t = t_steps[i].expand(shape[0])
        v = get_velocity_cfg(model, x, t, y, w_cfg)
        x = x + v * dt
        
    return x


@torch.no_grad()
def sample_heun(model, shape, device, y=None, steps=10, w_cfg=4.0):
    """
    [工业级升级] 二阶 Heun (Runge-Kutta 2) 积分器。
    大幅降低低步数 (如 4 步) 采样时的轨迹离散化截断误差。
    注意：每步需要评估两次网络，NFE 翻倍。
    """
    x = torch.randn(shape, device=device)
    t_steps = torch.linspace(0, 1, steps + 1, device=device)
    dt = 1.0 / steps
    
    for i in range(steps):
        t = t_steps[i].expand(shape[0])
        t_next = t_steps[i+1].expand(shape[0])
        
        # Step 1: 当前点评估速度
        v1 = get_velocity_cfg(model, x, t, y, w_cfg)
        x_next_euler = x + v1 * dt
        
        if i == steps - 1:
            # 最后一步退化为 Euler
            x = x_next_euler
        else:
            # Step 2: 目标点评估速度
            v2 = get_velocity_cfg(model, x_next_euler, t_next, y, w_cfg)
            # Heun 更新 (梯形求积)
            x = x + 0.5 * (v1 + v2) * dt
            
    return x

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str, default='checkpoints/latest.pt')
    parser.add_argument('--num_samples', type=int, default=16)
    parser.add_argument('--steps', type=int, default=10, help="积分步数, 越低越快")
    parser.add_argument('--image_size', type=int, default=64)
    parser.add_argument('--solver', type=str, choices=['euler', 'heun'], default='heun', help="常微分方程求解器选择")
    parser.add_argument('--w_cfg', type=float, default=4.0, help="CFG 引导强度")
    parser.add_argument('--target_class', type=int, default=0, help="条件生成的指定类别索引")
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = DiffusionTransformer(image_size=args.image_size, num_classes=1000).to(device)
    model.eval()
    
    try:
        model.load_state_dict(torch.load(args.ckpt, map_location=device))
        print(f"[*] 成功加载权重: {args.ckpt}")
    except Exception as e:
        print(f"[!] 加载权重失败，将使用随机初始化生成图像 ({e})")
        
    print(f"[*] 启动生成任务...")
    print(f"    生成参数: 样本={args.num_samples}, 步数={args.steps}, 求解器={args.solver}, CFG={args.w_cfg}, 类别={args.target_class}")
    
    shape = (args.num_samples, 3, args.image_size, args.image_size)
    y = torch.full((args.num_samples,), args.target_class, device=device, dtype=torch.long)
    
    if args.solver == 'heun':
        samples = sample_heun(model, shape, device, y=y, steps=args.steps, w_cfg=args.w_cfg)
        print(f"    [!] Heun 求解器使用了两次前向，实际 NFE: {args.steps * 2}")
    else:
        samples = sample_euler(model, shape, device, y=y, steps=args.steps, w_cfg=args.w_cfg)
        print(f"    [!] Euler 求解器实际 NFE: {args.steps}")
        
    samples = (samples + 1) / 2
    samples = torch.clamp(samples, 0, 1)
    
    torchvision.utils.save_image(samples, "samples.png", nrow=4)
    print("[*] 生成完毕！结果已保存至 samples.png")

if __name__ == "__main__":
    main()
