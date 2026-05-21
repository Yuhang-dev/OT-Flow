import argparse
import torch
import torchvision
from src.models.dit import DiffusionTransformer

@torch.no_grad()
def sample_euler(model, shape, device, steps=10):
    """
    使用常微分方程 (ODE) 的 Euler 积分器进行采样。
    Flow Matching 从 x0 (高斯噪声) 积分到 x1 (真实图像)。
    """
    model.eval()
    
    # 初始化 x0 为高斯噪声
    x = torch.randn(shape, device=device)
    
    # 时间步从 0 到 1
    t_steps = torch.linspace(0, 1, steps + 1, device=device)
    dt = 1.0 / steps
    
    for i in range(steps):
        t = t_steps[i].expand(shape[0])
        
        # 预测速度场 (Velocity Field)
        v = model(x, t)
        
        # Euler 步进: x_{t+dt} = x_t + v * dt
        x = x + v * dt
        
    return x

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str, default='checkpoints/latest.pt')
    parser.add_argument('--num_samples', type=int, default=16)
    parser.add_argument('--steps', type=int, default=10, help="Euler积分步数, 越低越快")
    parser.add_argument('--image_size', type=int, default=64)
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 初始化模型
    model = DiffusionTransformer(image_size=args.image_size).to(device)
    
    # 尝试加载权重
    try:
        model.load_state_dict(torch.load(args.ckpt, map_location=device))
        print(f"[*] 成功加载权重: {args.ckpt}")
    except Exception as e:
        print(f"[!] 加载权重失败，将使用随机初始化生成图像 ({e})")
        
    print(f"[*] 开始生成 {args.num_samples} 张图像, 步数={args.steps}...")
    shape = (args.num_samples, 3, args.image_size, args.image_size)
    
    samples = sample_euler(model, shape, device, steps=args.steps)
    
    # 将 [-1, 1] 还原为 [0, 1] 用于保存
    samples = (samples + 1) / 2
    samples = torch.clamp(samples, 0, 1)
    
    torchvision.utils.save_image(samples, "samples.png", nrow=4)
    print("[*] 生成完毕！结果已保存至 samples.png")

if __name__ == "__main__":
    main()
