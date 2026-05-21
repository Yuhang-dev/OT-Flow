import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

class ImageFolderDataset(Dataset):
    """
    一个简单的图片文件夹加载器。
    如果没有提供数据目录，则在测试模式下返回随机张量。
    """
    def __init__(self, data_dir=None, image_size=64):
        self.data_dir = data_dir
        self.image_size = image_size
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # 归一化到 [-1, 1]
        ])
        
        self.image_paths = []
        if data_dir and os.path.exists(data_dir):
            for root, _, files in os.walk(data_dir):
                for file in files:
                    if file.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
                        self.image_paths.append(os.path.join(root, file))
    
    def __len__(self):
        return len(self.image_paths) if len(self.image_paths) > 0 else 100 # 如果没数据，提供100个随机样本测试用
        
    def __getitem__(self, idx):
        if len(self.image_paths) == 0:
            # 返回 dummy data 测试
            return torch.randn(3, self.image_size, self.image_size)
            
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        return self.transform(image)
