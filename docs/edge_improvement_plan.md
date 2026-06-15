# Conv-Spatial EIT 边缘退化改进方案

## 问题背景

当前 ConvSpatialEIT 模型在物体靠近边界（距中心 >7cm）时重建精度显著下降：

| 距离 | RE 均值 | RE std | CC 均值 | CC std |
|------|---------|--------|---------|--------|
| 中心 (0-3cm) | **0.0825** | 0.0068 | **0.9881** | 0.0075 |
| 中部 (3-5cm) | 0.0880 | 0.0076 | 0.9867 | 0.0064 |
| 中远 (5-7cm) | 0.0982 | 0.0162 | 0.9835 | 0.0076 |
| **边缘 (7-10cm)** | **0.1284** | **0.0361** | **0.9486** | **0.0492** |

边缘 RE 比中心恶化 **55%**，CC 从 0.988 下降到 0.949。

---

## 根因分析

### 1. 训练数据不均衡

| 区域 | 单元数 | 占比 | 训练样本占比 |
|------|--------|------|------------|
| 中心环 (0-2cm) | 455 | 4.0% | ~10% |
| 边缘环 (8-10cm) | 4137 | 36.1% | **~8%** |

**原因**：`generate_circle_dataset.py` 中 `dist = uniform(0, max_dist)` 均匀采样圆心距，但边缘环面积更大（单元更多），同等样本数下边缘样本比例严重不足。

### 2. Conv2D 难以学习边缘的局域尖峰信号

- **中心物体** → ΔV 是宽缓平滑模式 → Conv2D 易学
- **边缘物体** → ΔV 是尖锐局域尖峰（集中在少数几个测量通道）→ Conv2D 难泛化

### 3. Jacobian 线性近似在边缘区域失效

| 指标 | 中心 | 边缘 |
|------|------|------|
| J列余弦相似度 | 0.693（高度相似） | 0.038（各单元独特） |
| 灵敏度强度 | 1x | 6.66x |
| 非线性度（尖峰率） | 2.30 | **18.37** (8x) |

**悖论**：边缘灵敏度是中心的 6.66 倍 → 传统算法效果好；但线性损失 ‖J·Δσ − ΔV‖² 在边缘区域误差高达 8 倍 → 无监督精调阶段反受其害。

### 4. Grid Sampling 分辨率不足

8×8 特征图经 grid_sample 插值到 11466 个单元，边缘单元密集（4137 个集中在 8-10cm 环带），相邻单元特征几乎相同，无法区分。

---

## 方案 A：数据层面改进

### A1. 数据生成平衡化

**目标**：使 50% 的训练样本位于边缘区域（圆心距 >5cm）

**改动文件**：`data/generate_circle_dataset.py`

```python
# 当前：均匀采样圆心距
dist = rng.uniform(0, max_dist)

# 改进：分区域按比例采样
if rng.random() < 0.5:
    # 中心区：圆心距 ∈ [0, 0.04]m
    dist = rng.uniform(0, 0.04)
else:
    # 边缘区：圆心距 ∈ [0.05, max_dist]m
    min_edge = min(0.05, max_dist * 0.6)
    dist = rng.uniform(min_edge, max_dist)
```

**预期效果**：边缘样本比例从 8% → 30-40%，RE 边缘估计可降至 0.10 以下。

### A2. 训练时分批分层采样

**目标**：每个 batch 确保至少 50% 边缘样本

**改动文件**：`train_conv_spatial.py`

```python
class BalancedBatchSampler:
    """
    分层采样器：每个 batch 一半边缘、一半中心
    """
    def __init__(self, dataset, batch_size, edge_threshold=0.05):
        self.labels = compute_edge_mask(dataset, threshold=edge_threshold)
        self.edge_idx = np.where(self.labels == 1)[0]
        self.center_idx = np.where(self.labels == 0)[0]
        self.batch_size = batch_size
        self.half = batch_size // 2

    def __iter__(self):
        np.random.shuffle(self.edge_idx)
        np.random.shuffle(self.center_idx)
        for i in range(0, min(len(self.edge_idx), len(self.center_idx)), self.half):
            batch = np.concatenate([
                self.edge_idx[i:i+self.half],
                self.center_idx[i:i+self.half]
            ])
            yield batch
```

---

## 方案 B：架构改进

### B1. 径向位置编码

**目标**：让模型显式知道每个单元离边界的距离

**改动文件**：`models/conv_spatial_eit.py`

在 ConvEncoder 的输出特征图上拼接待采样位置的归一化径向距离：

```python
class ConvSpatialEIT(nn.Module):
    def forward(self, x):
        # ... ConvEncoder 得到 feat (B, 128, 8, 8)

        # 计算每个 grid 采样点的径向距离
        # grid_norm: (1, n_elems, 2) 已归一化到 [-1, 1]
        radial_dist = torch.norm(grid_norm, dim=-1, keepdim=True)  # (1, n_elems, 1)

        # Grid Sampling
        sampled = F.grid_sample(feat, grid_norm, align_corners=False)  # (B, 128, n_elems, 1)
        sampled = sampled.squeeze(-1).permute(0, 2, 1)  # (B, n_elems, 128)

        # 拼接径向编码
        radial_enc = radial_dist.expand(x.size(0), -1, -1)  # (B, n_elems, 1)
        sampled = torch.cat([sampled, radial_enc], dim=-1)  # (B, n_elems, 129)

        # 调整 GNN 输入维度
        # GNN 输入从 128 → 129
```

**预期效果**：边缘单元获得额外的位置信号，帮助 GNN 区分密集的边界单元。

### B2. 多尺度扩张卷积

**目标**：同时捕捉宽缓（中心）和尖锐（边缘）的 ΔV 模式

**改动文件**：`models/conv_spatial_eit.py` 中 ConvEncoder

```python
class ConvEncoder(nn.Module):
    def __init__(self):
        # ...
        # 用扩张卷积替代部分普通卷积
        self.stage1 = nn.Sequential(
            ConvBlock(48, 96, kernel=3, dilation=1),  # 局部细节
            ConvBlock(96, 96, kernel=3, dilation=1),
        )
        self.stage2 = nn.Sequential(
            ConvBlock(96, 192, kernel=3, dilation=2, stride=2),  # 中等范围
            ConvBlock(192, 192, kernel=3, dilation=2),
        )
        self.stage3 = nn.Sequential(
            ConvBlock(192, 128, kernel=3, dilation=4, stride=2),  # 全局模式
            ConvBlock(128, 128, kernel=3, dilation=4),
        )
```

**预期效果**：扩张卷积增大感受野，使 Conv2D 能同时"看到"整个测量矩阵的宽缓模式和局部尖峰。

### B3. 边缘注意力模块

**目标**：在特征中显式增强边缘相关信息的权重

**改动文件**：`models/conv_spatial_eit.py`

在 GridSampler 之后、GNN 之前插入：

```python
class EdgeSpatialAttention(nn.Module):
    """
    可学习的边缘注意力：根据单元到边界的距离调整特征权重
    """
    def __init__(self, n_elems, hidden_dim):
        super().__init__()
        self.register_buffer('radial_dist',
            compute_normalized_radial_distance(centers, radius=0.1))
        self.gate = nn.Sequential(
            nn.Linear(1, hidden_dim // 4),
            nn.GELU(),
            nn.Linear(hidden_dim // 4, hidden_dim),
            nn.Sigmoid(),
        )

    def forward(self, features):
        # features: (B, n_elems, hidden_dim)
        weight = self.gate(self.radial_dist.unsqueeze(-1))  # (1, n_elems, hidden_dim)
        return features * (1.0 + weight)
```

**预期效果**：模型自动学习在边缘区域投入更多注意力资源。

---

## 方案 C：损失函数改进

### C1. 空间加权 MSE

**目标**：有监督预训练阶段，对边缘单元的预测误差施加更高权重

**改动文件**：`train_conv_spatial.py`

```python
# 预计算每个单元的权重
elem_centers = solver.element_centers[:, :2]
elem_dist = np.linalg.norm(elem_centers, axis=1)  # (n_elems,)
# 权重: 中心=1.0, 边缘=3.0 (线性映射)
weights = 1.0 + 2.0 * (elem_dist / DOMAIN_RADIUS)  # (n_elems,)
weights = torch.from_numpy(weights).float().to(device)

# 加权 MSE
loss = (weights * (out['sigma'] - S) ** 2).mean()
```

### C2. 全 FEM 无监督损失

**目标**：无监督精调阶段，用完整 FEM 求解替代 Jacobian 线性近似

**改动文件**：`train_conv_spatial.py`

```python
# 每 5 步运行一次完整 FEM
if step % fem_interval == 0:
    V_pred = solver.solve_multi_frequency(sigma_pred_np)
    loss_fem = F.mse_loss(V_pred, V_measured)
else:
    # 用缓存的 FEM 结果做线性近似
    loss_fem = F.mse_loss(V_cached + J @ delta_sigma, V_measured)
```

### C3. 边缘自适应 TV

**目标**：边缘区域用更弱的 TV 正则化（避免过度平滑）

```python
# TV 权重随距离递减
tv_weight_per_elem = 0.05 * (1.0 - 0.5 * elem_dist / DOMAIN_RADIUS)
# 中心: 0.05, 边缘: 0.025
loss_tv = (tv_weight_per_elem * tv_per_elem).mean()
```

---

## 方案 D：训练策略改进

### D1. 边缘课程学习

**目标**：先学容易的中心样本，逐步引入难的边缘样本

**改动文件**：`train_conv_spatial.py`

```python
for epoch in range(1, total_epochs + 1):
    if epoch < 10:
        edge_ratio = 0.0      # 前 10 epoch: 纯中心
    elif epoch < 20:
        edge_ratio = 0.2      # 10-20 epoch: 20% 边缘
    elif epoch < 30:
        edge_ratio = 0.35     # 20-30 epoch: 35% 边缘
    else:
        edge_ratio = 0.5      # 30+ epoch: 50% 边缘

    sampler.set_edge_ratio(edge_ratio)
```

### D2. 边缘数据增强

**目标**：对边缘样本做小幅扰动，扩充有限样本

```python
# 在数据生成时，对边缘样本额外做：
if dist > 0.05:  # 边缘样本
    # 圆心微扰
    cx += rng.uniform(-0.003, 0.003)
    cy += rng.uniform(-0.003, 0.003)
    # 半径微扰
    r *= rng.uniform(0.9, 1.1)
```

---

## 实施路线图

```
Phase 1: 数据平衡 (方案 A)
  ├── A1: 改 generate_circle_dataset.py
  └── A2: 改 train_conv_spatial.py (分层采样器)
  └── 验证: RE 边缘 < 0.105 ✓?

Phase 2: 损失改进 (方案 C)  
  ├── C1: 空间加权 MSE
  └── C3: 边缘自适应 TV
  └── 验证: RE 边缘 < 0.095 ✓?

Phase 3: 架构升级 (方案 B)
  ├── B1: 径向位置编码
  ├── B2: 多尺度卷积
  └── B3: 边缘注意力
  └── 验证: RE 边缘 < 0.085 ✓?

Phase 4: 训练策略 (方案 D)
  ├── D1: 课程学习
  └── D2: 数据增强
  └── 验证: RE 边缘 < 0.080 ✓?
```

---

## 预期最终效果

| 指标 | 当前 | Phase 1 | Phase 2 | Phase 3 | Phase 4 |
|------|------|---------|---------|---------|---------|
| 中心 RE | 0.0825 | 0.0820 | 0.0800 | 0.0780 | 0.0760 |
| 边缘 RE | **0.1284** | **0.1020** | **0.0920** | **0.0820** | **0.0760** |
| 整体 CC | 0.9835 | 0.9860 | 0.9880 | 0.9900 | 0.9920 |
