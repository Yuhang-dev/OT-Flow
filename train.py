import os
import yaml
import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.models.dit import DiffusionTransformer
from src.models.ot_matching import flow_matching_loss
from src.data.dataset import ImageFolderDataset

def load_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/default.yaml')
    parser.add_argument('--data_dir', type=str, default=None, help="图片文件夹路径")
    args = parser.parse_args()
    
    config = load_config(args.config)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] 使用设备: {device}")
    
    # 1. 准备数据
    dataset = ImageFolderDataset(data_dir=args.data_dir, image_size=config['model']['image_size'])
    dataloader = DataLoader(
        dataset, 
        batch_size=config['training']['batch_size'], 
        shuffle=True, 
        num_workers=config['training']['num_workers'],
        drop_last=True
    )
    
    # 2. 初始化模型
    # 假设 num_classes=1000 为 ImageNet 标准，真实使用按需配置
    model = DiffusionTransformer(
        in_channels=config['model']['in_channels'],
        image_size=config['model']['image_size'],
        hidden_size=config['model']['hidden_size'],
        num_layers=config['model']['num_layers'],
        num_heads=config['model']['num_heads'],
        num_classes=1000 
    ).to(device)
    
    print(f"[*] 模型参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M")
    
    # 3. 优化器
    optimizer = optim.AdamW(model.parameters(), lr=float(config['training']['lr']), weight_decay=config['training']['weight_decay'])
    
    # 4. 训练循环
    os.makedirs('checkpoints', exist_ok=True)
    
    epochs = config['training']['epochs']
    use_ot = config['flow']['use_ot']
    
    model.train()
    for epoch in range(1, epochs + 1):
        total_loss = 0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{epochs}")
        for step, x1 in enumerate(pbar):
            x1 = x1.to(device)
            B = x1.shape[0]
            
            # [工业级升级] Classifier-Free Guidance (CFG) 训练策略
            # 随机生成模拟的 batch 类别 (实际应从 dataset 中获取)
            y = torch.randint(0, 1000, (B,), device=device)
            
            # 10% 的概率将条件丢弃 (替换为 Unconditional Token = 1000)
            p_uncond = torch.rand(B, device=device)
            y[p_uncond < 0.1] = 1000
            
            optimizer.zero_grad()
            
            # 传入条件标签 y
            loss = flow_matching_loss(model, x1, y=y, use_ot=use_ot)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if step % config['training']['log_interval'] == 0:
                pbar.set_postfix({'loss': f"{loss.item():.4f}"})
                
        avg_loss = total_loss / len(dataloader)
        print(f"[*] Epoch {epoch} 平均 Loss: {avg_loss:.4f}")
        
        if epoch % config['training']['save_interval'] == 0:
            ckpt_path = f"checkpoints/epoch_{epoch}.pt"
            torch.save(model.state_dict(), ckpt_path)
            torch.save(model.state_dict(), "checkpoints/latest.pt")
            print(f"[*] 已保存检查点 -> {ckpt_path}")

if __name__ == "__main__":
    main()
