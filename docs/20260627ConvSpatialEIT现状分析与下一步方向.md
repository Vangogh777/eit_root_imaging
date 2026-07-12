# ConvSpatialEIT 现状分析与下一步方向

> 日期: 2026-06-27
> 模型: ConvSpatialEIT v2 (FrequencyCrossAttention + GATv2 + GridSampler)
> 现状: 监督训练 RE=0.10-0.11，无监督训练退化严重，DiffEIT 路线回归

---

## 1. 现状总览

### 1.1 最佳结果

| 指标 | 值 | 来源 |
|------|-----|------|
| **RE** | 0.1081 ± 0.030 | `validation_best_v2` (hd256) |
| **CC** | 0.9761 ± 0.022 | 同上 |
| **SSIM** | 0.9938 | 同上 |
| **PSNR** | 28.84 | 同上 |
| **IoU** | 0.928 | 同上 |
| **参数** | 6.3M (hd256) / 5.9M (hd512) | — |
| **推理** | 1.44ms/样本 | GPU |

历史最优 hd512 模型达到 **RE=0.102**（`validation_p1_fusion`），但后续 hd512 训练全部失败。

### 1.2 当前训练状态

⚠️ **ConvSpatialEIT 已停训 4 天**（最后成功训练: 2026-06-24）。最近 15+ 次训练全部是 DiffEIT v4，且结果不如 ConvSpatialEIT：

| 模型 | RE | 参数 | 状态 |
|------|-----|------|------|
| **ConvSpatialEIT v2 (hd256)** | **0.108** | 6.3M | ✅ 最佳 |
| ConvSpatialEIT v2 (hd512) | 0.102 (历史) | 5.9M | ❌ 最近全部训练失败 |
| DiffEIT v3 (conditional) | 0.331 | 38.8M | ❌ 差 3× |
| DiffEIT v4 (cosine, T=50) | 未成熟 | 38.8M | ❌ 训练中断 |

---

## 2. 核心问题诊断

### 问题 A: 无监督训练退化 [P0 🔴]

这是 ConvSpatialEIT 最大的问题：监督预训练 → 无监督微调后，RE 从 0.108 退化到 0.538（几乎输出均匀背景）。

**根因**：
1. Full FEM 每 10 步只在 4/32 样本上计算（12.5% 覆盖）
2. EIT 逆问题病态：边界电压不能唯一确定内部电导率，FEM 梯度方向可能是任意的
3. Measurement Consistency Loss 数值（~0.2-0.5）主导了 TV 正则（~0.025）
4. 当前半监督混合权重（0.8×MSE + 0.3×MCL）仍不够稳定

**证据**：`validation_final_v2` RE=0.538, CC≈0（无监督阶段输出）

### 问题 B: 网格空间分辨率瓶颈 [P1 🟡]

13×16 = 208 个 CNN grid 点 → 4424 个 FEM 元素，每个 grid 点平均覆盖 ~21 个元素。对于细粒度空间结构（如细长根、近边界含物）分辨率不足。

**证据**：边界附近样本 RE 显著偏高（worst cases RE > 0.2 多为含物靠近电极）

### 问题 C: 消融实验缺失 [P1 🟡]

不知道各组件的贡献：
- FrequencyCrossAttention 替换为 1×1 Conv 损失多少？
- GATv2 替换为 SimpleGNN 损失多少？
- 跳跃连接、位置编码、SE 通道注意力分别贡献多少？

### 问题 D: 数据多样性有限 [P1 🟡]

- 训练数据以单圆/简单形状为主
- 对比度固定 5×（真实场景变化大）
- 多内含物、近边界、复杂形状场景覆盖不足
- 已有 4 个测试集（test / low_noise / high_noise / near_boundary）但训练数据未见系统性匹配

### 问题 E: hd512 训练不稳定 [P2 🟢]

历史最佳 RE=0.102 是 hd512 模型，但最近 7 次 hd512 训练全部失败（fail）。Degraded to hd256 (RE=0.108)。

---

## 3. 架构现状确认

ConvSpatialEIT v2 的组件清单：

| 组件 | 状态 | 效果 |
|------|------|------|
| FrequencyCrossAttention (6频融合) | ✅ 已实现 | 替换简单取第1频，预期提升显著 |
| GATv2 (4头注意力 + 边特征) | ✅ 已实现 | 替换 SimpleGNN sum 聚合 |
| ConvEncoder + 跳跃连接 | ✅ 已实现 | 基础编码 |
| SE 通道注意力 | ✅ 已实现 | 通道维度自适应 |
| GridSampler (双线性插值) | ✅ 已实现 | CNN→GNN 桥梁 |
| Fourier 位置编码 + 半径编码 | ✅ 已实现 | 空间位置信息 |
| Jacobian 分支 (Jᵀr 残差) | ⚠️ 可选 | `--use_model_jacobian`，默认关闭 |
| VoltageMasking (20% 随机mask) | ⚠️ 可选 | 数据增强 |
| 两阶段训练 | ⚠️ 第二阶段退化 | 监督→无监督，第二阶段无效 |
| Full FEM MCL | ⚠️ 不稳定 | fem_interval=10, subset=4 |

---

## 4. 下一步方向

### 路线总览

```
ConvSpatialEIT v2 → v2.1 (快速修复) → v2.2 (架构优化) → v3 (论文级)
         RE=0.10            RE=0.08             RE=0.06            RE=0.05
         6.3M              6.3M                7-8M               8-10M
```

### Phase 1: v2.1 快速修复（1-2天，预期 RE 0.10→0.08）

**目标**：在不改架构的前提下，修复训练流程，稳定达到 RE~0.08

#### 1.1 废弃无监督第二阶段，纯监督训练

当前无监督阶段不仅无益反而有害。建议：
```python
# 方案 A: 纯监督训练更长
--mode supervised --epochs_sup 150  # 替代 80+200 两阶段

# 方案 B: 极弱半监督锚点
total = 0.95 * MSE(sigma_pred, sigma_gt) + 0.05 * physical_loss
# 保留物理约束信号但极小权重，不会导致退化
```

**工作量**: 1 行代码改动  
**风险**: 低  
**收益**: 稳定达到 RE~0.10，可能超越到 ~0.08

#### 1.2 恢复 hd512 训练稳定性

hd512 历史上达到 RE=0.102（优于 hd256），但最近 7 次训练全部失败。需要诊断失败原因：
- 检查 OOM？（GATv2 + hd512 = 更大的 attention 矩阵）
- 梯度爆炸？（GATv2 多层可能导致）
- Batch size / grad_accum 配置问题？

**建议**:
```bash
python train_conv_spatial.py \
  --hidden_dim 512 --gnn_layers 4 --use_gat \
  --batch_size 8 --grad_accum_steps 4 \
  --mode supervised --epochs_sup 150 \
  --lr 1e-4
```

**工作量**: 1-2 次训练运行调试

#### 1.3 数据增强：训练集多样性匹配测试集

已有 4 个测试集，但训练数据未系统性覆盖这些场景：
- test_near_boundary: 近边界含物 → 训练加入 20% 近边界样本
- test_high_noise (-15dB): 高噪声 → 训练加入高噪声样本
- 对比度变化: 3×, 5×, 8× 混合训练

**工作量**: 修改 `data/generate_mixed_dataset.py`

---

### Phase 2: v2.2 架构优化（3-5天，预期 RE 0.08→0.06）

#### 2.1 网格分辨率翻倍

```
当前: (B, 832, 13, 16) → GridSampler → (B, 4424, 832)
改进: (B, 832, 13, 16) → ConvTranspose2d → (B, 256, 26, 32)
      → GridSampler → (B, 4424, 256)
```

效果：208 → 832 个 grid 点，每个点只覆盖 ~5 个元素（原来是 ~21 个）
增加参数 ~0.3M

#### 2.2 FrequencyCrossAttention 维度对齐

当前存在维度不匹配：
- FrequencyCrossAttention: d_model=64（硬编码）
- ConvEncoder: base_ch=96 → 192 → 384 → 256
- GNN: hidden=256

GNN 接收的输入是 832+35=867d（来自 13×16 网格的 832 维特征）。但 FrequencyCrossAttention 只输出 d_model=64 的特征。

**建议**: 统一维度设计
```python
# 使 FrequencyCrossAttention 输出与 ConvEncoder 基础通道对齐
d_model = base_ch  # 96, 让 attention 输出直接进入 stem
```

#### 2.3 Jacobian 分支默认开启并增强

当前 Jacobian 分支可选且默认关闭。分析显示其有益：
- Jᵀr 反向投影提供了明确的物理信息
- 门控机制比简单拼接更有效

**建议**:
```python
# 默认开启，使用门控融合
gate = Sigmoid(Linear(g))  # g = normalized Jᵀr
sigma = sigma_0 + gate * delta
```

#### 2.4 系统消融实验

运行以下消融，量化每个组件的贡献：

| Ablation | 修改方式 | 预期 RE |
|----------|---------|---------|
| Baseline (完整模型) | — | 0.10 |
| w/o FrequencyCrossAttention | 替换为 1×1 Conv | ~0.12-0.14 |
| w/o GATv2 | use_gat=False → SimpleGNN | ~0.12-0.13 |
| w/o 位置编码 | 移除 Fourier PE + 半径编码 | ~0.11-0.12 |
| w/o 跳跃连接 | 只保留主路径 (256d) | ~0.12-0.14 |
| w/o SE 注意力 | 移除 SEModule | ~0.10-0.11 |
| w/ Jacobian 分支 | 开启 use_model_jacobian | ~0.09-0.10 |

---

### Phase 3: v3 论文级（1-2周）

#### 3.1 不确定性估计
添加 Gaussian NLL 输出头，提供像素级置信度：
```python
loss = ||σ_pred - σ_gt||² / exp(logvar) + logvar
```

#### 3.2 传统方法系统对比
- BP (Back Projection)
- GN (Gauss-Newton) 
- GREIT
- 不同噪声水平下的系统对比

#### 3.3 真实数据适配
- 植物根系真实截面
- 不同容器形状（方桶、椭圆桶）
- 不同电极数（8/16/32）

#### 3.4 推理优化
- ONNX 导出
- TensorRT 加速
- 批量化推理 API

---

## 5. 建议立即执行（今天/明天）

### 优先级排序

| # | 任务 | 预期效果 | 时间 | 风险 |
|---|------|---------|------|------|
| **1** | **纯监督训练 150 epoch** | RE 稳定 0.10 | 3-4h | 低 |
| **2** | **修复 hd512 训练** | RE → 0.08-0.09 | 半天 | 中 |
| **3** | **消融: w/o FreqAttn** | 量化最大组件的贡献 | 1h* | 低 |
| **4** | **网格分辨率翻倍** | 边界精度提升 | 1天 | 低 |
| **5** | **Jacobian 分支默认开启** | 物理先验增强 | 2h | 低 |

*注: 消融实验只需修改一行代码然后运行 evaluate，不需要重新训练（除非要做完整的训练消融）

### 不推荐继续的方向

- ❌ **DiffEIT 继续训练**: 38.8M 参数模型 RE=0.331，比 6.3M 的 ConvSpatialEIT RE=0.108 差 3×。扩散模型在 EIT 上的价值存疑。
- ❌ **无监督第二阶段（当前形式）**: 已被多次实验证明会导致退化。除非有全新的物理约束方案（如 adversial physics loss, multi-scale FEM），否则不值得继续尝试。

---

## 6. 参考

- 最佳 checkpoint: `checkpoints/conv_spatial_best.pt` (106MB)
- 训练配置: `config/train_config.yaml`
- 评估脚本: `evaluate_conv_spatial_v3.py`
- 完整分析历史: `docs/20260624ConvSpatialEIT现状以及改进.md`
- 瓶颈分析: `docs/network_bottleneck_analysis.md`
