# OT-Flow: Optimal Transport Guided Flow Matching with DiT

本项目实现了一个基于**流匹配 (Flow Matching)** 和 **最优传输 (Optimal Transport, OT)** 的轻量化条件扩散模型。

与标准的 Diffusion Model 相比，OT-Flow 通过匈牙利算法在批次内寻找数据与噪声的最优耦合，使得生成的概率流轨迹更加平直，从而大幅加快训练收敛速度，并能在极少步数（如 10 步）内生成高质量图像。

## 项目结构
```text
OT-Flow/
├── train.py                  # 完整的主训练脚本 (带混合精度与 Checkpoint 保存)
├── requirements.txt          # 依赖清单
├── README.md                 # 项目说明
├── configs/
│   └── default.yaml          # 训练与模型配置
├── src/
│   ├── models/
│   │   ├── dit.py            # Diffusion Transformer (DiT) 架构
│   │   └── ot_matching.py    # OT 耦合与 Flow Matching Loss 核心逻辑
│   └── data/
│       └── dataset.py        # 数据集加载器 (支持通用图像文件夹格式)
└── scripts/
    └── sample.py             # Euler 积分采样器，用于生成图像
```

## 快速开始

### 1. 环境配置
```bash
pip install -r requirements.txt
```

### 2. 训练模型
```bash
python train.py --config configs/default.yaml --data_dir /path/to/your/image/folder
```

### 3. 推理生成图像
```bash
python scripts/sample.py --ckpt checkpoints/latest.pt --num_samples 16 --steps 10
```

## 技术亮点
- **OT-Coupling**: 使用 `scipy.optimize.linear_sum_assignment` 实现无偏的确定性最优传输匹配。
- **Flow Matching**: 抛弃传统扩散模型的 DDPM 噪声预测，直接使用回归损失 (MSE) 拟合 Velocity Field (速度场)。
- **DiT Backbone**: 采用可拓展的 Transformer 主干，支持 Patch 化处理。
