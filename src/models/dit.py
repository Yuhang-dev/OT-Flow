import torch
import torch.nn as nn
import math

def timestep_embedding(timesteps, dim, max_period=10000):
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding

class DiTBlock(nn.Module):
    def __init__(self, hidden_size, num_heads):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size)
        self.attn = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Linear(hidden_size * 4, hidden_size)
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size)
        )

    def forward(self, x, c_emb):
        # x: [B, Seq, D], c_emb (time + label): [B, D]
        shift, scale = self.adaLN_modulation(c_emb).chunk(2, dim=-1)
        # AdaLN
        x_norm = self.norm1(x) * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        # Attention
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + attn_out
        # MLP
        x = x + self.mlp(self.norm2(x))
        return x

class DiffusionTransformer(nn.Module):
    """
    [工业级升级] 支持 CFG (Classifier-Free Guidance) 的 DiT 模型。
    """
    def __init__(self, in_channels=3, image_size=64, hidden_size=384, num_layers=6, num_heads=6, num_classes=1000):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        
        # Patch 嵌入
        self.patch_size = 4
        self.seq_len = (image_size // self.patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, hidden_size, kernel_size=self.patch_size, stride=self.patch_size)
        
        # 位置编码
        self.pos_embed = nn.Parameter(torch.zeros(1, self.seq_len, hidden_size))
        
        # 时间步嵌入 MLP
        self.time_embed = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size)
        )
        
        # [工业级升级] 类别条件嵌入 (额外 +1 用于 Unconditional 空标签)
        self.label_emb = nn.Embedding(num_classes + 1, hidden_size)
        
        # DiT Blocks
        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads) for _ in range(num_layers)
        ])
        
        # 输出映射
        self.norm_final = nn.LayerNorm(hidden_size)
        self.out_proj = nn.Linear(hidden_size, self.patch_size * self.patch_size * in_channels)
        
    def forward(self, x, t, y=None):
        B, C, H, W = x.shape
        # Patchify
        x = self.proj(x) # [B, D, H/p, W/p]
        x = x.flatten(2).transpose(1, 2) # [B, Seq, D]
        
        # 添加位置编码
        x = x + self.pos_embed
        
        # 时间特征
        t_emb = timestep_embedding(t, self.hidden_size)
        t_emb = self.time_embed(t_emb) # [B, D]
        
        # [工业级升级] 融合条件标签
        if y is None:
            # 推理时如果没传 y，默认全用 unconditional
            y = torch.full((B,), self.num_classes, device=x.device, dtype=torch.long)
            
        y_emb = self.label_emb(y) # [B, D]
        
        # 将时间和标签特征相加 (或 Concat)，注入 AdaLN
        c_emb = t_emb + y_emb
        
        # 经过所有 Block
        for block in self.blocks:
            x = block(x, c_emb)
            
        x = self.norm_final(x)
        x = self.out_proj(x) # [B, Seq, p*p*C]
        
        # 还原回图像形状 Unpatchify
        p = self.patch_size
        h_out = H // p
        w_out = W // p
        x = x.view(B, h_out, w_out, p, p, C).permute(0, 5, 1, 3, 2, 4).reshape(B, C, H, W)
        return x
