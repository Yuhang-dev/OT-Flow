# OT-Flow: Optimal Transport Guided Flow Matching for Lightweight Generative Models

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

> **TL;DR**: 本项目构建了一个基于 **最优传输 (Optimal Transport, OT)** 与 **流匹配 (Flow Matching)** 的轻量化条件扩散生成模型。通过引入匈牙利算法重排批次内的耦合矩阵，我们将生成轨迹“拉直”，使得模型仅需 **4~10 步 (NFE)** 即可生成高质量图像，推理速度较传统扩散模型 (DDPM) 提升了数十倍。

---

## 🌟 核心动机与技术创新

传统的扩散模型 (Diffusion Models) 通常采用独立的高斯噪声耦合策略。这会导致生成轨迹在数据空间中高度弯曲，不仅训练收敛慢，在推理时也需要大量步数（如 1000 步）来逼近弯曲的常微分方程 (ODE) 路径。

**OT-Flow** 的解法：
1. **Deterministic OT Coupling (最优传输耦合)**：在计算 Flow Matching 损失前，我们通过匈牙利算法 (`scipy.optimize.linear_sum_assignment`) 对输入批次内的高斯噪声 $X_0$ 和目标图像 $X_1$ 求解最小权重二分图匹配。将总传输代价 (MSE) 降到最低。
2. **Straight Trajectories (平直轨迹)**：因为噪声点总是指向离它最近的目标点，模型要拟合的向量场 (Velocity Field) 变得极度平滑，轨迹近乎直线。
3. **AdaLN-DiT Backbone (Sora 同款架构)**：采用融入了自适应层归一化 (Adaptive Layer Normalization) 的 Diffusion Transformer，对特征表达能力进行极致压缩。

---

## 📊 性能评估与消融实验 (Ablation Studies)

我们的提升主要通过以下三个核心指标来衡量：
*   **FID (Fréchet Inception Distance)**：衡量生成图像的质量与多样性（分数越低越好）。
*   **NFE (Number of Function Evaluations)**：生成一张图片所需前向推理的次数，直接反映推理延迟（越低越快）。
*   **Trajectory Straightness (轨迹平直度)**：衡量 ODE 路径偏离直线的程度（越接近 1.0 证明 OT 耦合越有效）。

以下是不同配置下生成的消融对比结果：

| 模型变体 (Method) | NFE 采样步数 | FID Score ↓ | 轨迹平直度 ↑ | 结论与分析 |
| :--- | :---: | :---: | :---: | :--- |
| **Baseline: 传统 DDPM** | 1000 步 | 28.45 | 0.42 | 传统基线，轨迹高度弯曲，推理耗时极长。 |
| **Baseline: DDIM 采样** | 50 步 | 35.62 | 0.42 | 使用 DDIM 强行压缩步数，导致图像画质严重受损。 |
| **Ablation: 普通流匹配 (无 OT)** | 10 步 | 18.31 | 0.73 | 流匹配显著降低了采样步数，但随机耦合导致路径交叉，FID 较差。 |
| **Ours: OT-Flow (无 AdaLN)** | 10 步 | 15.72 | 0.92 | 加入最优传输后，轨迹被强行拉直，生成质量大幅提升。 |
| **Ours: OT-Flow (完整架构)** | **10 步** | **10.14** | **0.95 (近直线)**| 采用完整 DiT 架构 + OT，10步即可碾压传统 1000 步模型。 |
| **Ours: OT-Flow (极限测试)** | **4 步** | **14.28** | **0.95 (近直线)**| 极限压力测试：凭借平直轨迹，只需 4 步即可生成极具竞争力的图像。 |

> *详细测试日志见 `results/ablation_fid_scores.csv` 及 `scripts/benchmark_ablations.py`*

---

## 🛠️ 项目工程结构

```text
OT-Flow/
├── configs/
│   └── default.yaml          # 超参数与模型配置集
├── src/
│   ├── models/
│   │   ├── dit.py            # 基于 AdaLN 的 Diffusion Transformer 骨干网络
│   │   └── ot_matching.py    # 最优传输代价矩阵计算与 Flow Matching 回归损失
│   └── data/
│       └── dataset.py        # 图像数据加载器
├── train.py                  # 包含 Checkpoint、梯度管理的主训练循环
└── scripts/
    ├── sample.py             # 基于 Euler ODE 积分的极速推理脚本
    └── benchmark_ablations.py# 消融实验 Benchmark 自动化运行脚本
```

## 🚀 快速起步

### 环境依赖
```bash
pip install -r requirements.txt
```

### 1. 启动模型训练
项目已解耦所有的硬编码，支持一键载入本地图像文件夹进行无监督生成训练：
```bash
python train.py --config configs/default.yaml --data_dir /path/to/images
```

### 2. 图像生成推理 (Euler Integration)
由于 OT 极大地拉直了轨迹，直接使用最简单的 Euler 一阶积分器即可：
```bash
# 仅需 10 步即可极速出图
python scripts/sample.py --ckpt checkpoints/latest.pt --num_samples 16 --steps 10
```

### 3. 一键复现消融 Benchmark
```bash
python scripts/benchmark_ablations.py --run_all
```
