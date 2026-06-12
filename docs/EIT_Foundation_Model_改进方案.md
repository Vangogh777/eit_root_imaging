# EIT Foundation Model 改进方案

## 1. 问题分析

### 1.1 当前系统现状

| 项目 | 当前状态 | 问题 |
|------|---------|------|
| 模型架构 | SimpleSFSBLC (MLP) | 容量有限，缺乏空间建模能力 |
| 训练方式 | 无监督学习 | 无真实标签约束，收敛困难 |
| 数据量 | 100-1000 样本 | 训练样本不足 |
| 精度 | RE ~0.42 | 相对误差较高 |

### 1.2 精度瓶颈分析

```
当前流程：边界电压 V → MLP → 电导率 σ

问题：
1. MLP 无法建模空间关系（网格单元之间的邻接关系）
2. 无监督损失函数优化空间大，容易陷入局部最优
3. 缺乏先验知识注入（物理规律、典型根结构）
4. 数据量不足导致泛化能力差
```

---

## 2. 统一架构设计

### 2.1 整体架构：多模态 EIT Foundation Model

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    Multi-Modal EIT Foundation Model                      │
│                         (多模态EIT基础模型)                               │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐               │
│  │   测量数据    │    │   物理约束    │    │  文本描述    │               │
│  │  (边界电压)   │    │  (Jacobian)  │    │ (LLM先验)   │               │
│  │  Input: V    │    │  Input: J    │    │ Input: Text  │               │
│  └──────┬───────┘    └──────┬───────┘    └──────┬───────┘               │
│         │                   │                   │                       │
│         ▼                   ▼                   ▼                       │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐               │
│  │  ViT/Swin    │    │  Physics     │    │    LLM       │               │
│  │  Encoder     │    │  Encoder     │    │   Encoder    │               │
│  │  (视觉编码)   │    │  (物理编码)   │    │  (文本编码)   │               │
│  └──────┬───────┘    └──────┬───────┘    └──────┬───────┘               │
│         │                   │                   │                       │
│         │    特征维度:      │    特征维度:      │    特征维度:           │
│         │    (B, N, D)      │    (B, M, D)      │    (B, L, D)          │
│         │                   │                   │                       │
│         └───────────────────┼───────────────────┘                       │
│                             ▼                                           │
│                    ┌──────────────────┐                                 │
│                    │   Cross-         │                                 │
│                    │   Attention      │                                 │
│                    │   Fusion Module  │                                 │
│                    │  (多模态融合)     │                                 │
│                    └────────┬─────────┘                                 │
│                             │                                           │
│                             ▼                                           │
│                    ┌──────────────────┐                                 │
│                    │   Diffusion      │                                 │
│                    │   Decoder        │                                 │
│                    │   (生成式重建)    │                                 │
│                    └────────┬─────────┘                                 │
│                             │                                           │
│                             ▼                                           │
│                    ┌──────────────────┐                                 │
│                    │   电导率分布 σ    │                                 │
│                    │   (重建结果)      │                                 │
│                    └──────────────────┘                                 │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### 2.2 各模块设计

#### 2.2.1 视觉编码器 (ViT/Swin Encoder)

**设计思路**：将测量数据重构为"虚拟图像"，利用预训练视觉模型的特征提取能力。

```
输入: 边界电压 V (B, n_freq, n_meas)
      ↓
重构: 将电压映射到2D网格空间
      V → V_grid (B, n_freq, H, W)
      ↓
ViT编码: 使用预训练ViT提取特征
      V_grid → features_vit (B, N_patches, D)
      ↓
输出: 视觉特征 (B, N, D)
```

**实现要点**：
- 使用预训练 ViT-B/16 或 Swin-Tiny
- 测量数据 → 2D 极坐标网格映射
- 可选：冻结预训练权重，仅训练投影层

#### 2.2.2 物理编码器 (Physics Encoder)

**设计思路**：编码物理约束（Jacobian矩阵、正问题模型），为重建提供物理先验。

```
输入: Jacobian矩阵 J (n_freq, n_meas, n_elems)
      ↓
图神经网络: 将网格建模为图结构
      nodes = 网格单元中心
      edges = 单元邻接关系
      ↓
GNN编码: Graph Attention Network
      J → features_phys (B, M, D)
      ↓
输出: 物理特征 (B, M, D)
```

**实现要点**：
- 使用 Graph Attention Network (GAT)
- 输入包括：Jacobian、单元中心坐标、邻接矩阵
- 物理约束作为注意力偏置

#### 2.2.3 文本编码器 (LLM Encoder)

**设计思路**：利用LLM理解物理知识，生成文本先验描述。

```
输入: 测量描述文本
      "16电极圆形域，相邻激励，检测到低频高阻抗区域..."
      ↓
LLM编码: 使用预训练LLM
      text → features_llm (B, L, D)
      ↓
输出: 文本特征 (B, L, D)
```

**实现要点**：
- 使用轻量级LLM（如 Qwen-1.8B、Phi-2）
- 可以用规则模板自动生成测量描述
- 可选：使用 CLIP 风格的文本编码器

#### 2.2.4 多模态融合模块 (Cross-Attention Fusion)

**设计思路**：使用交叉注意力机制融合多模态特征。

```
输入:
  - 视觉特征 F_v (B, N, D)
  - 物理特征 F_p (B, M, D)
  - 文本特征 F_t (B, L, D)
      ↓
Cross-Attention:
  Q = F_v (主查询)
  K, V = concat(F_p, F_t) (键值对)
      ↓
Multi-Head Attention:
  Fused = Attention(Q, K, V) + F_v
      ↓
输出: 融合特征 (B, N, D)
```

**实现要点**：
- 多层 Transformer Block
- 层归一化 + 残差连接
- 可学习的模态权重

#### 2.2.5 Diffusion 解码器

**设计思路**：使用扩散模型逐步生成电导率分布。

```
训练阶段 (加噪):
  σ_0 → σ_1 → ... → σ_T (纯噪声)

去噪过程:
  σ_T → σ_{T-1} → ... → σ_0 (重建结果)

条件生成:
  条件 = 融合特征 F_fused
  预测噪声 ε_θ(σ_t, t, F_fused)
```

**实现要点**：
- 使用 DDPM 或 DDIM 采样器
- 条件注入：Cross-Attention 或 AdaLN
- 时间步编码：Sinusoidal Position Encoding

---

## 3. 三阶段训练策略

### 3.1 Stage 1: 有监督预训练

**目标**：在大规模模拟数据上学习 EIT 重建的基本能力。

```
数据: 100,000+ 模拟样本
      - 随机根结构 (taproot/fibrous/herringbone)
      - 多频率测量
      - 真实标签 σ_gt

模型: 仅使用 ViT Encoder + 简单Decoder
      (不使用 Diffusion，加速训练)

损失: L1 + L2 + Perceptual Loss
      L = ||σ_pred - σ_gt||_1 + ||σ_pred - σ_gt||_2 + L_perc

预期效果: RE < 0.15
```

### 3.2 Stage 2: 多模态对齐

**目标**：对齐视觉、物理、文本三个模态的特征空间。

```
数据: 50,000+ 样本 + 物理约束 + 文本描述

模型: 完整架构（不含Diffusion）

损失: 对比学习 + 重建损失
      L = L_recon + λ_1 * L_contrastive + λ_2 * L_physics

      L_contrastive: 同一样本的多模态特征应该接近
      L_physics: ||F(σ_pred) - V_measured||^2

预期效果: RE < 0.10
```

### 3.3 Stage 3: 领域微调

**目标**：适应真实植物根成像场景。

```
数据:
  - 真实测量数据（如有）
  - 高保真模拟数据

模型: 完整架构 + Diffusion Decoder

损失: Diffusion Loss + Physics Loss
      L = L_diffusion + λ * L_physics

预期效果: RE < 0.08 (模拟数据)
```

---

## 4. 实现计划

### 4.1 目录结构

```
eit_foundation/
├── models/
│   ├── vision/
│   │   ├── vit_encoder.py        # ViT编码器
│   │   ├── swin_encoder.py       # Swin编码器
│   │   └── voltage_to_image.py   # 电压→图像映射
│   ├── physics/
│   │   ├── gnn_encoder.py        # 图神经网络编码器
│   │   └── jacobian_encoder.py   # Jacobian编码器
│   ├── text/
│   │   ├── llm_encoder.py        # LLM文本编码器
│   │   └── text_templates.py     # 文本模板生成
│   ├── fusion/
│   │   └── cross_attention.py    # 交叉注意力融合
│   ├── diffusion/
│   │   ├── unet.py               # U-Net骨干
│   │   ├── ddpm.py               # DDPM采样器
│   │   └── conditional.py        # 条件注入
│   └── foundation.py             # 整体模型
├── training/
│   ├── stage1_pretrain.py        # 阶段1训练
│   ├── stage2_align.py           # 阶段2训练
│   ├── stage3_finetune.py        # 阶段3训练
│   └── losses.py                 # 损失函数
├── data/
│   ├── generate_large.py         # 大规模数据生成
│   ├── text_generator.py         # 文本描述生成
│   └── dataset.py                # 数据集类
├── evaluation/
│   ├── metrics.py                # 评估指标
│   └── visualize.py              # 可视化
├── configs/
│   ├── stage1.yaml               # 阶段1配置
│   ├── stage2.yaml               # 阶段2配置
│   └── stage3.yaml               # 阶段3配置
└── scripts/
    ├── train.sh                  # 训练脚本
    └── eval.sh                   # 评估脚本
```

### 4.2 开发时间线

| 阶段 | 任务 | 时间 | 优先级 |
|------|------|------|--------|
| **Week 1-2** | Stage 1: 数据生成 + ViT模型 | 2周 | P0 |
| **Week 3-4** | Stage 1: 有监督训练 + 调优 | 2周 | P0 |
| **Week 5-6** | Stage 2: 物理编码器 + 多模态融合 | 2周 | P1 |
| **Week 7-8** | Stage 2: 对比学习训练 | 2周 | P1 |
| **Week 9-10** | Stage 3: Diffusion Decoder | 2周 | P2 |
| **Week 11-12** | Stage 3: 端到端微调 | 2周 | P2 |

### 4.3 硬件需求

| 阶段 | GPU需求 | 预计训练时间 |
|------|---------|-------------|
| Stage 1 | RTX 3090 / A100 (24GB+) | 2-3天 |
| Stage 2 | A100 (40GB+) | 3-5天 |
| Stage 3 | A100 (40GB+) | 5-7天 |

---

## 5. 预期效果

### 5.1 精度提升预测

| 方案 | 预期RE | 相对提升 |
|------|--------|---------|
| 当前基线 | 0.42 | - |
| Stage 1 (有监督+ViT) | < 0.15 | +64% |
| Stage 2 (多模态融合) | < 0.10 | +76% |
| Stage 3 (Diffusion) | < 0.08 | +81% |

### 5.2 能力扩展

1. **不确定性量化**：Diffusion模型可生成多个可能结果
2. **可解释性**：多模态注意力可视化
3. **泛化能力**：预训练模型可迁移到其他EIT场景
4. **少样本学习**：Foundation Model 可快速适应新场景

---

## 6. 风险与备选方案

### 6.1 主要风险

| 风险 | 影响 | 缓解措施 |
|------|------|---------|
| 数据量不足 | 模型欠拟合 | 数据增强、迁移学习 |
| Diffusion训练不稳定 | 收敛困难 | 使用预训练权重、渐进式训练 |
| 计算资源不足 | 训练时间过长 | 使用轻量级模型、梯度累积 |
| 模拟数据与真实数据差异大 | 泛化能力差 | Domain Adaptation、对抗训练 |

### 6.2 备选方案

1. **轻量级方案**：仅使用 Stage 1，省略 Diffusion
2. **增量方案**：在现有模型基础上逐步添加模块
3. **预训练方案**：使用现成的 Foundation Model（如 SAM、CLIP）

---

## 7. 总结

本方案提出了一个统一的多模态 EIT Foundation Model 架构，融合了：

1. **视觉大模型 (ViT/Swin)**：强大的空间特征提取能力
2. **物理编码器 (GNN)**：显式建模物理约束
3. **LLM 文本编码器**：注入先验知识
4. **Diffusion 解码器**：生成式重建，提升精度

通过三阶段训练策略，预期将重建精度从 RE=0.42 提升到 RE<0.08，同时获得不确定性量化、可解释性等额外能力。

---

## 附录：参考文献

1. Dosovitskiy et al. "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale" (ViT, 2020)
2. Liu et al. "Swin Transformer: Hierarchical Vision Transformer using Shifted Windows" (2021)
3. Ho et al. "Denoising Diffusion Probabilistic Models" (DDPM, 2020)
4. Hamilton et al. "Inductive Representation Learning on Large Graphs" (GraphSAGE, 2017)
5. Radford et al. "Learning Transferable Visual Models From Natural Language Supervision" (CLIP, 2021)
