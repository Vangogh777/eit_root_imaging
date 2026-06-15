# RE 精度提升方案 — ConvSpatialEIT

> 当前最佳 RE ≈ 0.1152（有监督预训练 epoch 16），之后停滞在 0.1-0.2 之间波动。
> 本文档分析瓶颈并给出可落地的优化方案，按性价比排序。

---

## 1. 瓶颈分析

### 1.1 数据瓶颈（最严重）

| 问题 | 现状 | 影响 |
|------|------|------|
| **形状单一** | 只有单圆内含物 | 模型学的是"找圆"而非通用电导率重建 |
| **对比度固定** | 始终 0.01→0.05 (5x) | 无法泛化到不同对比度的真实场景 |
| **大小变化有限** | 半径 0.8~3cm | 未见过的尺寸效果差 |
| **信噪比固定** | -40~-20dB | 缺乏噪声多样性 |

### 1.2 模型瓶颈

| 问题 | 现状 | 影响 |
|------|------|------|
| **GNN 聚合简单** | 仅 sum 聚合，无注意力 | 等权对待所有邻居，丢失关键边信息 |
| **无跳跃连接** | ConvEncoder 输出直连 GNN | 高分辨率信息无法传递到输出层 |
| **编码器通道** | base_ch=48，输出 128 维 | 容量偏小，特征表达能力不足 |

### 1.3 训练瓶颈

| 问题 | 现状 | 影响 |
|------|------|------|
| **无 EMA** | 直接使用当前权重 | 权重波动大，错过最佳点 |
| **无数据增强** | 原样输入电压 | 泛化性差 |
| **LR 调度简单** | 单一 CosineAnnealing | 缺少 warmup 和重启 |
| **MSE 等权** | 所有单元同等重要 | 边界过渡区域精度低 |

---

## 2. 优化方案

### 2.1 数据多样性（收益最大，改动最小）

**目标**：替换 `generate_circle_dataset.py`，增加形状多样性。

```python
# 新增形状类型（随机选择）:
# A. 单圆          — 沿用现有逻辑
# B. 多圆 (2-4个)  — 随机位置、大小
# C. 椭圆          — 随机长短轴、旋转角
# D. 环形          — 圆环，随机内外径
# E. 不规则凸包    — 随机顶点生成的凸多边形
```

**参数范围扩展**：

| 参数 | 当前 | 扩展后 |
|------|------|--------|
| 形状类型 | 单圆 | 圆/椭圆/多圆/环/多边形 |
| 对比度 | 5x (0.05/0.01) | 2x~15x (0.02~0.15) |
| 内含物尺寸 | 0.8~3cm | 0.5~4.5cm |
| 噪声范围 | -40~-20dB | -50~-15dB |
| 背景不均 | 均匀 0.01 | 可加梯度/随机波动 |

**预计收益**：RE 从 0.115 降至 **0.08-0.09**，且泛化性显著提升。

### 2.2 训练增强（纯策略改动，不改模型）

**2.2.1 EMA（指数移动平均）**

```python
# 训练时维护一份平滑权重，验证/推理时使用
ema_model = torch.optim.swa_utils.AveragedModel(model)
ema_model.update_parameters(model)  # 每步更新

# 验证时用 ema_model
# 保存时保存 ema 权重
```

**预计收益**：RE 提升 **0.01-0.02**，几乎零成本。

**2.2.2 电压通道随机掩码**

```python
# 训练时随机掩码部分测量通道（数据增强）
mask = torch.rand(B, 1, n_meas) > 0.1  # 10% 通道置零
V = V * mask.to(device)
```

EITDataset 已有 `voltage_mask_ratio` 参数，只需启用。

**预计收益**：RE 提升 **0.005-0.01**，零成本。

**2.2.3 Cosine Annealing with Warm Restarts**

```python
# 每个重启周期重新加热，跳出局部最优
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer, T_0=20, T_mult=2  # 首周期 20 epoch，后续翻倍
)
```

**预计收益**：RE 提升 **0.01**，改动一行代码。

### 2.3 损失函数精调（可控改动）

**2.3.1 边缘加权 MSE**

```python
# 过渡区域（sigma 接近 0.02-0.03）赋予更高权重
edge_mask = (gt > 0.012) & (gt < 0.045)  # 过渡区
weights = torch.where(edge_mask, 3.0, 1.0)  # 过渡区权重 3x
loss = (weights * (pred - gt).pow(2)).mean()
```

**预计收益**：边界精度提升，RE 改善 **0.005-0.01**。

**2.3.2 SSIM 结构损失（可选）**

```python
# 不只在像素级比较，还在结构级比较
from torchmetrics import StructuralSimilarityIndexMeasure
ssim = StructuralSimilarityIndexMeasure(data_range=0.1)
loss = 0.8 * mse_loss + 0.2 * (1 - ssim(pred_2d, gt_2d))
```

**预计收益**：视觉质量提升明显，RE 改善 **0.005**。

### 2.4 模型架构改进（改动较大，可选）

**2.4.1 注意力 GNN 层**

```python
# 当前: sum 聚合 (x_agg = scatter_add(x_src * w))
# 改为: 注意力聚合 (学到的边权重)
class AttnGNNLayer(SimpleGNNLayer):
    def forward(self, x, edge_idx, edge_weight):
        # 计算注意力系数
        h_src = self.attn_q(x[:, src])  # query
        h_dst = self.attn_k(x[:, dst])  # key
        attn = F.softmax((h_src * h_dst).sum(-1), dim=-1)
        x_agg = scatter_add(x[:, src] * attn.unsqueeze(-1))
        return MLP(cat([x, x_agg]))
```

**预计收益**：表达力提升，RE 改善 **0.01-0.02**，但显存增加 ~20%。

**2.4.2 跳跃连接（U-Net 风格）**

```python
# 在 ConvEncoder 中保留多级特征
feat_list = [x1, x2, x3]  # 不同分辨率特征

# GNN 输出后融合编码器特征
h = gnn_output
h = cat([h, upsample(feat_list[-1])])  # 跳跃连接
h = MLP(h)
```

**预计收益**：高分辨率信息保留，RE 改善 **0.01**。

---

## 3. 实施路线图

### 阶段 1（立即做，2-3 小时）

- [ ] 数据生成器增加多形状：椭圆、多圆、环形
- [ ] 对比度/尺寸/噪声范围扩展
- [ ] 重新生成数据集
- [ ] 训练验证 RE 是否下降

### 阶段 2（同一天内）

- [ ] 添加 EMA 模型平滑
- [ ] 启用电压通道随机掩码增强
- [ ] 切换至 CosineAnnealingWarmRestarts
- [ ] 添加边缘加权 MSE 损失

### 阶段 3（可选，后续迭代）

- [ ] 注意力 GNN 层
- [ ] U-Net 跳跃连接
- [ ] SSIM 结构损失

---

## 4. 预期效果

| 优化项 | 预计 RE 改善 | 累计 RE |
|--------|:----------:|:------:|
| 当前 | — | ~0.115 |
| + 数据多样性 | 0.03-0.04 ↓ | **~0.08** |
| + EMA + 数据增强 | 0.02 ↓ | **~0.06** |
| + 损失精调 | 0.01 ↓ | **~0.05** |
| + 模型架构改进 | 0.01-0.02 ↓ | **~0.04** |

> RE=0.05 意味着重建电导率与真实值的平均相对误差约 5%，在 EIT 领域属于高精度水平。
> 物理约束无监督精调后还有进一步提升空间。

---

## 5. 验证方法

每项优化实施后，用以下流程验证：

```bash
# 1. 重新训练（有监督 20 epoch）
python train_conv_spatial.py --epochs_sup 20 --epochs_unsup 0 --batch_size 64

# 2. 评估
python evaluate_conv_spatial.py --checkpoint checkpoints/conv_spatial_best.pt --split test

# 3. 对比 RE 和 CC 指标，以及可视化结果
```

如果连续 3 个 epoch RE 没有再创新低，即可确认达到了当前配置的精度上限。
