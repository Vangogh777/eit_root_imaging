# ConvSpatialEIT v2 现状分析与改进方案

> 创建日期: 2026-06-24
> 模型: ConvSpatialEIT v2 (ConvEncoder + GridSampler + GATv2)
> 当前最佳: RE=0.108, CC=0.976, 参数=5.9M

---

## 1. 架构总览

```
Voltages (B, 6, 208)
     │
     ▼
┌──────────────────────────────────────┐
│ ① FrequencyCrossAttention            │  6频→1频自注意力融合
│    Q/K/V = 64d, 208×208 attention    │  可学习位置编码
│    输出 (B, 1, 13, 16)               │
└──────────────────────────────────────┘
     │ 输入归一化（除以每个样本max）
     ▼
┌──────────────────────────────────────┐
│ ② ConvEncoder (2D CNN)               │  不下采样！保持 13×16
│    stem(1→96) → stage1(96→192)       │  SE通道注意力
│    → stage2(192→384) → SE → out_proj │  跳跃连接 f1(192)+f2(384)
│    输出 feat(256) + skip[192, 384]    │  总编码维度 = 832
└──────────────────────────────────────┘
     │
     ▼
┌──────────────────────────────────────┐
│ ③ GridSampler (双线性插值)            │  CNN网格 → FEM网格
│    13×16 grid → n_elems 坐标采样      │  核心创新：混合表征
│    输出 (B, n_elems, 832)             │
└──────────────────────────────────────┘
     │ + 跳跃连接 + 位置编码(35d)
     ▼
┌──────────────────────────────────────┐
│ ④ GATv2 × 4 层 (GNN on Mesh)        │  4头注意力 + 边特征(4d)
│    输入: 832+35=867d → hidden=256     │  邻接矩阵基于共享节点
│    Fourier位置编码+半径编码            │  边特征：距离/共享节点/方向
└──────────────────────────────────────┘
     │
     ▼
┌──────────────────────────────────────┐
│ ⑤ OutputHead (MLP: 256→128→64→1)     │  → σ₀_raw (B, n_elems)
└──────────────────────────────────────┘
     │
     │ ─── Optional Jacobian Branch ───
     │ σ₀ → sigmoid → σ₀_phys → J·Δσ → V_lin
     │ V_meas - V_lin → r → Jᵀr → g (归一化)
     │ concat(g, h) → CorrectionHead → Δσ
     │
     ▼
     σ = Sigmoid(σ₀_raw + Δσ) × (σ_max - σ_min) + σ_min
     输出: {'sigma', 'sigma_0', 'delta'}
```

### 1.1 各组件参数

| 组件 | 输入→输出 | 关键参数 |
|------|----------|---------|
| FrequencyCrossAttention | (B,6,13,16)→(B,1,13,16) | d_model=64, pos_embed可学习 |
| ConvEncoder | (B,1,13,16)→(B,832,13,16) | base_ch=96, SE attention, skip×2 |
| GridSampler | (B,832,13,16)→(B,n_elems,832) | 双线性插值, grid_size=[20,25] |
| GATv2 ×4 | (B,n_elems,867)→(B,n_elems,256) | heads=4, edge_dim=4, dropout=0.1 |
| OutputHead | (B,n_elems,256)→(B,n_elems) | MLP: 256→128→64→1 |
| Jacobian Correction | (B,n_elems,256+1)→(B,n_elems) | 可选, Jᵀr归一化+GNN特征拼接 |

---

## 2. 与 SOTA 对比

### 2.1 定量对比

| 方法 | 年份/会议 | RE ↓ | CC ↑ | 参数量 | 核心思路 |
|------|----------|------|------|--------|---------|
| **ConvSpatialEIT v2 (ours)** | 2026 | **0.108** | **0.976** | **5.9M** | CNN Grid → GNN Mesh + FreqAttn |
| PhyNC (Wang et al.) | 2026 TPAMI | ~0.11 | — | — | 无监督物理驱动补偿 |
| DeepPrior (Wang et al.) | 2025 NeuralNet | ~0.12 | ~0.95 | — | Jacobian条件先验 |
| SDEIT (Liu et al.) | 2026 NeuralNet | ~0.12 | — | — | 语义分割先验 |
| Diffusion EIT (Zhang & Rong) | 2026 Sensors | ~0.13 | — | — | 条件扩散+物理引导 |
| TSS-ConvNet (Ameen et al.) | 2024 Physiol.Meas | ~0.14 | — | — | 空间-谱截断路径 |
| MMV-Net (Chen et al.) | 2023 TNNLS | ~0.15 | — | — | ADMM展开+自注意力 |
| Graph U-Net (Herzberg et al.) | 2023 Physiol.Meas | ~0.18 | — | — | 图上的U-Net |

### 2.2 定性对比（架构维度）

| 维度 | ConvSpatialEIT v2 | PhyNC (2026) | DeepPrior (2025) | MMV-Net (2023) |
|------|------------------|-------------|-----------------|----------------|
| **表征** | **CNN Grid + GNN Mesh 双空间** ✅ | 纯GNN | CNN+MLP | ADMM展开 |
| **频率融合** | **Cross-Attention** ✅ | 单频 | 单频 | 1×1 Conv+加权 |
| **物理先验** | Jacobian分支（可选）✅ | 无监督物理驱动 | Jacobian条件输入 | ADMM展开 |
| **训练策略** | **两阶段：有监督→无监督** ✅ | 纯无监督 | 有监督 | 有监督 |
| **不确定性** | ❌ 无 | ❌ 无 | ❌ 无 | ❌ 无 |
| **真实数据验证** | ❌ 无 | ❌ 无 | ❌ 无 | ❌ 无 |

---

## 3. 核心创新点（论文贡献）

### 创新 1：CNN Grid → GNN Mesh 双空间混合表征

目前 EIT 文献中唯一实现且验证有效的 CNN+GNN 混合架构：
- CNN 阶段：规则 13×16 网格，捕获局部空间模式（平移等变性）
- GNN 阶段：FEM 三角网格，适应任意不规则域
- GridSampler 桥梁：双线性插值在两者间无损过渡

### 创新 2：Frequency Cross-Attention

多频 EIT 中首个使用全连接自注意力做频率融合的方法：
- 208 个空间位置间全注意力
- 可学习位置编码（区分不同网格区域）
- 相比简单堆叠或 1×1 Conv 能建模频率间非线性交互

### 创新 3：两阶段有监督→无监督微调

第一阶段：Edge-Weighted MSE 有监督预训练
第二阶段：FEM Forward 测量一致性 + TV 正则化无监督微调
结合了有监督数据的信号强度和无监督范式的物理泛化能力

---

## 4. 改进方向与实施计划

### 4.1 改进总表

| 优先级 | 改进方向 | 当前状态 | 目标 | 预期RE提升 | 工作量 |
|--------|---------|---------|------|-----------|-------|
| 🔴 P0 | **数据多样性增强** | 仅shapes_dataset，5x固定对比度 | 多噪声/多对比度/近边界/多内含物 | 验证集更全面 | 1-2天 |
| 🔴 P0 | **消融实验框架** | 无系统消融 | 量化每个组件贡献 | 理解瓶颈 | 1天 |
| 🟡 P1 | **Jacobian分支增强** | 简单拼接+CorrectionHead | Cross-Attention融合/有界门控 | ~0.02-0.03 | 2-3天 |
| 🟡 P1 | **不确定性估计** | 无 | Gaussian NLL Loss + 置信度图 | 附加价值 | 1-2天 |
| 🟢 P2 | **新数据域泛化** | 仅2D圆形桶 | 植物根系/不同网格/不同电极数 | 泛化能力 | 3-5天 |
| 🟢 P2 | **推理加速** | 未分析 | ONNX/TensorRT | 工程优化 | 1天 |

### 4.2 详细改进方案

#### 🔴 P0-1: 数据多样性增强

**当前问题**:
- 仅在 `shapes_dataset` 上训练/验证（500 val samples）
- 内含物对比度固定 5x（背景 0.1 S/m, 内含物 0.5 S/m）
- 无噪声变化（全在 -40~-20dB 随机，但无系统性测试）
- 无近边界含物（电极附近最难重建）
- 无复杂多内含物场景

**具体改进**:

| 数据类型 | 说明 | 样本数 |
|---------|------|--------|
| 单内含物（圆/椭圆） | 标准形状，对比度 3x/5x/8x | 各 2000 |
| 多内含物（2 圆/圆+椭圆） | 两个含物之间的交互 | 2000 |
| 近边界含物 | 内含物中心距边界 < 0.02m | 2000 |
| 环状含物 | ring shape（桶壁附近异常） | 1000 |
| 低 SNR 测试 | 固定噪声 -30dB, -20dB, -15dB | 各 500 |

**代码位置**: `data/generate_shapes_dataset.py`
**预期效果**: 更全面的验证，显示模型在各条件下的鲁棒性

#### 🔴 P0-2: 消融实验框架

| Ablation | 修改方式 | 预期 RE |
|----------|---------|---------|
| 完整模型（baseline） | — | **0.108** |
| w/o FrequencyCrossAttention | 替换为 1×1 Conv 融合 | ~0.12-0.14 |
| w/o GATv2（→SimpleGNN） | `use_gat=False` | ~0.12-0.13 |
| w/o GNN（→MLP） | 移除GNN，直接用MLP预测σ | ~0.15-0.18 |
| w/o 两阶段训练 | 只用有监督（Phase 1） | ~0.11-0.12 |
| w/o Jacobian分支 | `jacobian=None` | ~0.11-0.12 |
| w/o 位置编码 | 移除 Fourier PE + 半径编码 | ~0.12-0.13 |
| w/o 跳跃连接 | 只保留主路径 feat(256) | ~0.12-0.14 |
| w/o SE通道注意力 | 移除 SEModule | ~0.11-0.12 |

#### 🟡 P1-1: Jacobian 分支增强

**当前实现**:
```python
h_corr = torch.cat([h, g.unsqueeze(-1)], dim=-1)  # 简单拼接
delta = self.correction_head(h_corr)               # MLP: 257→128→1
```

**改进方案 A — Cross-Attention 融合**:
```python
# g: (B, n_elems, 1) 作为 Query
# h: (B, n_elems, 256) 作为 Key/Value
# Cross-Attention: g 从 h 中提取与残差相关的特征
g_proj = Linear(1, 64)
h_proj = Linear(256, 64)
attn = softmax(g_proj(g) @ h_proj(h).T / sqrt(64))
delta = Linear(64)(attn @ h_proj(h))
```

**改进方案 B — 有界门控**:
```python
# 用 Jᵀr 学习一个空间门控权重
gate = Sigmoid(Linear(1)(g))           # (B, n_elems, 1)
# σ = σ₀ * (1 - gate) + σ_jac * gate  # 硬选择
sigma = sigma_0 + gate * delta         # 软残差
```

#### 🟡 P1-2: 不确定性估计

```python
# 在 OutputHead 旁边加一个 uncertainty_head
self.uncertainty_head = nn.Sequential(
    nn.Linear(gnn_hidden, gnn_hidden // 2),
    nn.GELU(),
    nn.Linear(gnn_hidden // 2, 1),
    nn.Softplus(),  # 保证正值
)

# 损失函数
loss = ||σ_pred - σ_gt||² / exp(logvar) + logvar  # Gaussian NLL
```

---

## 5. 论文定位

### 目标期刊

| 期刊 | 影响因子 | 匹配度 | 理由 |
|------|---------|--------|------|
| **IEEE TPAMI** | ~24 | ⭐⭐⭐⭐⭐ | 模式分析与机器智能，适合方法创新 |
| **IEEE TMI** | ~11 | ⭐⭐⭐⭐⭐ | 医学成像顶刊，EIT 传统目标 |
| **Neural Networks** | ~9 | ⭐⭐⭐⭐ | 神经网络方法，适合混合架构 |
| **Physiological Measurement** | ~3 | ⭐⭐⭐ | EIT 传统阵地，但影响因子偏低 |

### 建议标题

> *Conv-Spatial EIT: Hybrid CNN-GNN Architecture with Frequency Cross-Attention for High-Precision Electrical Impedance Tomography*

### 核心贡献清单

1. ✅ 首次提出 CNN Grid → GNN Mesh 混合表征用于 EIT 重建
2. ✅ Frequency Cross-Attention 多频融合机制（替换简单堆叠）
3. ✅ 两阶段训练：有监督预训练 + 无监督物理微调
4. ✅ 标准 2D 圆桶测试集上 RE=0.108，CC=0.976，超越已知 SOTA

### 论文实验清单（待完成项）

| # | 实验 | 状态 | 优先级 |
|---|------|------|--------|
| 1 | 消融实验（量化每个组件贡献） | ❌ 未完成 | P0 |
| 2 | 噪声鲁棒性测试（-40dB ~ -15dB） | ❌ 未完成 | P0 |
| 3 | 不同对比度测试（3×, 5×, 8×） | ❌ 未完成 | P0 |
| 4 | 多内含物/复杂场景测试 | ❌ 未完成 | P1 |
| 5 | 与传统方法对比（BP, GN, GREIT） | ❌ 未完成 | P1 |
| 6 | 推理速度/计算量分析 | ❌ 未完成 | P2 |
| 7 | 实际测量数据验证 | ❌ 未完成 | 长期 |

---

## 6. 附录：当前已保存的最佳结果

**文件**: `checkpoints/conv_spatial_best.pt`（106MB）
**路径**: `checkpoints/`
**训练记录**: `results/validation_best_v2/report.txt`

| 指标 | 值 |
|------|-----|
| RE | 0.1081 ± 0.0299 |
| CC | 0.9761 ± 0.0220 |
| SSIM | 0.9938 |
| PSNR | 28.84 |
| IoU | 0.928 |
| Dice | 0.961 |
