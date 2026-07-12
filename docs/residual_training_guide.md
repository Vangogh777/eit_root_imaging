# ResidualEIT 训练方案说明

> 日期: 2026-06-23
> 训练脚本: `train_residual_eit.py`

---

## 📋 训练模式分析

### 当前配置（弱监督模式）

根据 `config/residual_eit_config.yaml` 的配置：

```yaml
loss_weights:
  supervised: 1.0              # 有监督损失权重
  residual_measurement: 1.0    # 残差测量一致性
  tv: 0.03                     # TV正则
  delta_l1: 0.01               # Δσ稀疏约束
  delta_smooth: 0.02           # Δσ平滑约束
```

**训练模式**：**弱监督（Weakly Supervised）**

---

## 🎯 训练模式对比

### 模式 A: 完全有监督

```yaml
loss_weights:
  supervised: 1.0
  residual_measurement: 0.0    # 关闭物理约束
  tv: 0.0
  delta_l1: 0.0
  delta_smooth: 0.0
```

**特点**：
- ✅ 训练最快，收敛最稳定
- ✅ 可以验证"残差架构"本身的能力
- ❌ 需要真实 σ 标签
- ❌ 可能过拟合仿真数据

**适用场景**：验证架构上限，快速原型测试

### 模式 B: 弱监督（当前配置）⭐ 推荐

```yaml
loss_weights:
  supervised: 1.0              # 有监督锚点
  residual_measurement: 1.0    # 物理约束
  tv: 0.03                     # 空间平滑
  delta_l1: 0.01               # 稀疏约束
  delta_smooth: 0.02           # 图平滑
```

**特点**：
- ✅ 结合监督信号和物理约束
- ✅ 平衡训练稳定性和物理合理性
- ✅ 防止过拟合
- ✅ 提升泛化能力

**适用场景**：主要训练模式，兼顾性能和泛化

### 模式 C: 无监督

```yaml
loss_weights:
  supervised: 0.0              # 关闭监督信号
  residual_measurement: 1.0    # 纯物理约束
  tv: 0.05
  delta_l1: 0.02
  delta_smooth: 0.02
```

**特点**：
- ✅ 不需要真实 σ 标签
- ✅ 完全依赖物理约束
- ❌ 训练难度大，收敛慢
- ❌ 需要精确的 Jacobian 和残差计算

**适用场景**：真实数据训练，无标签场景

---

## 📊 当前训练配置详解

### 1. 模型配置

```yaml
model:
  name: "residual_eit"
  hidden_dim: 256              # GNN 隐藏层维度
  gnn_layers: 4                # GNN 层数
  dropout: 0.1
  use_gat: true                # 使用 GATv2
  n_heads: 4                   # 注意力头数
  sigma_min: 0.005             # 电导率下限
  sigma_max: 0.1               # 电导率上限
  delta_scale: 0.02            # Δσ 缩放因子
```

**参数说明**：
- `delta_scale=0.02`：限制 Δσ ∈ [-0.02, 0.02]，防止过修正
- `hidden_dim=256`：与 ConvSpatialEIT hd256 对比
- `gnn_layers=4`：适中的网络深度

### 2. 损失函数组合

**总损失公式**：
```python
L_total = 1.0 * L_sup              # 有监督锚点
        + 1.0 * L_residual_meas    # 残差测量一致性
        + 0.03 * L_tv              # TV正则
        + 0.01 * L_delta_l1        # Δσ稀疏
        + 0.02 * L_delta_smooth    # Δσ平滑
```

**各项损失的作用**：

| 损失项 | 作用 | 权重 | 来源 |
|--------|------|------|------|
| `L_sup` | 让 σ̂ 接近真实 σ | 1.0 | 有标签数据 |
| `L_residual_meas` | 让 J·Δσ ≈ r（物理约束） | 1.0 | 物理原理 |
| `L_tv` | 最终 σ 的空间平滑 | 0.03 | 正则化 |
| `L_delta_l1` | Δσ 稀疏性 | 0.01 | 防止过修正 |
| `L_delta_smooth` | Δσ 在图上的平滑 | 0.02 | 空间连续性 |

### 3. 训练超参数

```yaml
training:
  batch_size: 16
  val_batch_size: 32
  epochs: 100
  learning_rate: 3.0e-4
  eta_min: 1.0e-6
  weight_decay: 1.0e-5
  grad_clip: 1.0
```

**说明**：
- 学习率 3e-4：中等学习率，适合弱监督训练
- 梯度裁剪 1.0：防止梯度爆炸
- 100 epochs：足够收敛

---

## 🔄 训练流程

### 数据流程

```
HDF5 数据集 (已有预计算特征):
  ├── voltages: (N, 6, 208)         # 边界电压
  ├── sigmas: (N, n_elems)          # 真实电导率
  ├── sigma_0: (N, n_elems)         # BP传统重建
  ├── physics_g: (N, n_elems)       # Jᵀr 物理特征
  └── voltage_residual: (N, 208)    # 电压残差 r

训练时:
  1. 加载 batch: (V, σ_gt, σ₀, g, r)
  2. 前向传播: Δσ = MeshGNN(σ₀, g, pe, z_v)
  3. 重建: σ̂ = σ₀ + Δσ
  4. 计算损失:
     - L_sup = ||σ̂ - σ_gt|| / ||σ_gt||
     - L_residual_meas = ||J·Δσ - r||²
     - L_tv = TV(σ̂)
     - L_delta_l1 = |Δσ|₁
     - L_delta_smooth = mean((Δσ_i - Δσ_j)²)
  5. 反向传播 + 梯度裁剪
  6. 优化器更新
```

### 评估指标

训练过程中会计算：
- **train_loss**: 训练总损失
- **val_loss**: 验证总损失
- **val_RE**: 验证相对误差 `||σ̂ - σ_gt|| / ||σ_gt||`
- **val_coarse_RE**: 传统重建 RE `||σ₀ - σ_gt|| / ||σ_gt||`

**关键指标对比**：
- 如果 `val_RE < val_coarse_RE`：说明神经网络成功改进了传统重建
- 改善幅度 = (coarse_RE - val_RE) / coarse_RE

---

## 🚀 启动训练

### 方案 1: 默认配置（推荐）

```bash
# 使用弱监督模式
python train_residual_eit.py
```

### 方案 2: 无监督模式

```bash
# 修改配置文件，将 supervised 设为 0.0
python train_residual_eit.py
```

### 方案 3: 纯监督模式（快速验证）

```bash
# 修改配置文件，关闭物理约束
python train_residual_eit.py
```

### 方案 4: 自定义参数

```bash
# 直接修改 config/residual_eit_config.yaml
# 然后运行
python train_residual_eit.py --config config/residual_eit_config.yaml
```

---

## 📈 预期效果

### 与 ConvSpatialEIT 对比

| 指标 | ConvSpatialEIT | ResidualEIT (预期) | 改善 |
|------|:--------------:|:------------------:|:----:|
| **RE** | 0.193 | ≤ 0.15 | ↓ 20-25% |
| **CC** | 0.955 | ≥ 0.97 | ↑ 1.5% |
| **物理可解释性** | 低 | 高 | ⬆️⬆️⬆️ |
| **训练稳定性** | 中 | 高 | ⬆️⬆️ |

### 与传统重建对比

| 方法 | RE | 说明 |
|------|:--:|------|
| **BP 传统重建** | ~0.25-0.30 | σ₀ 基线 |
| **ResidualEIT** | ≤ 0.15 | σ₀ + Δσ 改进 |
| **改善幅度** | ~40-50% | 神经网络贡献 |

---

## ⚠️ 注意事项

### 1. 数据集要求

✅ **当前数据集已满足**：
- `sigma_0`: BP 传统重建结果
- `physics_g`: Jᵀr 物理特征
- `voltage_residual`: 电压残差

### 2. Jacobian 文件

必须存在：`data/generated/jacobian.npy`

当前已有：
```bash
ls -lh data/generated/jacobian.npy
# -rw-r--r-- 1 ubuntu ubuntu 55M Jun 15 16:18 jacobian.npy
```

### 3. GPU 内存

预计使用：
- batch_size=16: ~4-6GB
- batch_size=32: ~8-10GB

当前 GPU: RTX 4090 (24GB) ✅ 充足

---

## 📊 训练监控

### 关键观察指标

1. **RE 下降趋势**：
   - 应该持续下降
   - 如果不降，检查数据或损失权重

2. **coarse_RE vs val_RE**：
   - val_RE 应该 < coarse_RE
   - 否则网络没有学到有用的东西

3. **损失平衡**：
   - L_sup 和 L_residual_meas 应该平衡
   - 如果一方过大，调整权重

4. **Δσ 统计**：
   - |Δσ| 应该较小（在 delta_scale 范围内）
   - 过大说明网络过度修正

### 可视化建议

```python
# 训练后可视化对比
import matplotlib.pyplot as plt

# 对比：传统重建 vs 神经修正
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(sigma_0)      # BP 传统重建
axes[1].imshow(sigma_pred)   # ResidualEIT
axes[2].imshow(sigma_gt)     # 真实
plt.show()
```

---

## 🎯 推荐训练方案

### 阶段 1: 验证架构（1-2 天）

```yaml
# 弱监督模式，当前配置
loss_weights:
  supervised: 1.0
  residual_measurement: 1.0
  tv: 0.03
  delta_l1: 0.01
  delta_smooth: 0.02
```

**目标**：验证 ResidualEIT 能超过 ConvSpatialEIT

### 阶段 2: 消融实验（2-3 天）

对比不同配置：
- 纯监督 vs 弱监督 vs 无监督
- 不同损失权重的影响
- delta_scale 的影响

### 阶段 3: 超参数优化（3-5 天）

调优：
- 学习率
- batch_size
- GNN 层数
- hidden_dim

---

## 💡 立即行动

**推荐：使用当前弱监督配置开始训练**

```bash
# 1. 检查数据
python -c "import h5py; f=h5py.File('data/generated/eit_dataset.h5','r'); print('数据集OK')"

# 2. 启动训练（tmux 保持会话）
tmux new -s residual_training
python train_residual_eit.py

# 3. 监控训练
# 在另一个终端
tail -f nohup.out  # 如果用 nohup
# 或在 tmux 中直接观察
```

**预期训练时间**：
- 单个 epoch: ~1-2 分钟（100个样本，batch=16）
- 总时间: ~100-200 分钟（100 epochs）
- **约 2-3 小时**

---

## 📝 总结

### 当前配置特点

✅ **弱监督模式**：结合监督信号和物理约束
✅ **数据就绪**：所有预计算特征已存在
✅ **参数合理**：基于 Codex 建议的默认配置
✅ **GPU 充足**：RTX 4090 完全够用

### 预期效果

🎯 **RE ≤ 0.15**：相比当前最佳 RE=0.193 改善 20-25%
🎯 **CC ≥ 0.97**：结构相似性提升
🎯 **可解释性提升**：物理引导的残差修正

### 立即启动

```bash
python train_residual_eit.py
```

**预计 2-3 小时后看到结果！**
