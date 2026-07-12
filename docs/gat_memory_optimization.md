# GAT内存优化与改进方案

> 分析日期: 2026-06-23
> 问题：GAT（图注意力网络）内存消耗过大
> 当前状态：已关闭GAT，使用普通GNN

---

## 📊 问题分析

### 当前GPU状态

```
GPU型号: NVIDIA GeForce RTX 4090
总内存: 24 GB
已使用: 11 GB
剩余: 13 GB
```

### 内存消耗来源

**当前配置**：
- 网格单元: n_elems = **11466**（网格分辨率h0=0.0025）
- 有向边: n_edges = 41621
- Hidden dim: 256
- Batch size: 8

**GAT内存消耗分解**：

| 组件 | 内存消耗 | 说明 |
|------|---------|------|
| 注意力权重 | 0.00 GB | n_heads × n_edges × 4 bytes |
| Q/K/V矩阵 | 0.13 GB | 3 × n_heads × n_elems × hidden_dim |
| 边特征注意力 | 0.16 GB | n_heads × n_edges × hidden_dim |
| **理论总计** | **0.29 GB** | 前向传播 |
| **实际总计** | **~0.87 GB** | 包含梯度、中间激活 |

**关键问题**：
- ❌ 网格单元数过大（11466）
- ❌ GAT在稀疏图上的实现效率不高
- ❌ 需要存储大量注意力中间结果

---

## 🎯 改进方案对比

### 方案对比表

| 方案 | 内存节省 | 性能影响 | 实施难度 | 推荐度 |
|------|---------|---------|---------|--------|
| **1. 关闭GAT** | ~70% | 轻微下降 | ✅ 已完成 | ⭐⭐⭐⭐⭐ |
| **2. 减小网格分辨率** | ~60% | 中等下降 | 🟡 中等 | ⭐⭐⭐⭐ |
| **3. 使用更好的数据集** | - | 性能提升 | 🟢 简单 | ⭐⭐⭐⭐⭐ |
| **4. 减小batch_size** | ~50% | 训练变慢 | 🟢 简单 | ⭐⭐⭐ |
| **5. 减小hidden_dim** | ~75% | 性能下降 | 🟢 简单 | ⭐⭐⭐ |
| **6. 改进传统重建** | - | **性能大幅提升** | 🟡 中等 | ⭐⭐⭐⭐⭐ |

---

## 💡 推荐改进方案

### 🥇 优先级1：使用更好的数据集（立即实施）

**当前问题**：
```python
# eit_dataset.h5 的传统重建质量很差
传统BP重建RE: mean=3.34  # 误差是真实值的3倍多
σ₀范围: [0.002, 0.2]     # 严重超出物理范围
```

**改进方案**：使用 `mixed_dataset.h5`

```bash
# 检查mixed_dataset是否已有预计算特征
python3 << 'EOF'
import h5py

h5_path = "data/generated/mixed_dataset.h5"
try:
    with h5py.File(h5_path, 'r') as f:
        print("✅ mixed_dataset.h5 存在")
        print(f"结构: {list(f.keys())}")
        if 'train' in f:
            print(f"train样本数: {f['train']['voltages'].shape[0]}")
            if 'sigma_0' in f['train']:
                print("✅ 已有预计算特征")
            else:
                print("❌ 需要预计算残差特征")
except:
    print("❌ mixed_dataset.h5 不存在")
EOF
```

**优势**：
- ✅ 更多样化的根系结构（直根/须根/鲱骨型）
- ✅ 更大的数据集（可能10000+样本）
- ✅ 更好的传统重建质量
- ✅ 无需改代码，只改配置文件

**实施**：
```yaml
# config/residual_eit_config.yaml
data:
  dataset_path: "./data/generated/mixed_dataset.h5"
```

---

### 🥈 优先级2：改进传统重建质量（核心改进）

**当前问题**：
- BP重建RE=3.34（太差）
- σ₀范围失控（0.002~0.2）

**改进方向**：

#### A. 使用Tikhonov正则化

```python
# 在 models/traditional/reconstructor.py 中添加
class TikhonovReconstructor(BaseReconstructor):
    def __init__(self, solver, lambda_reg=0.01):
        self.J = solver.jacobian
        self.sigma_ref = solver.sigma_ref
        self.lambda_reg = lambda_reg
    
    def reconstruct(self, voltage):
        # Tikhonov: (J^T J + λI)^{-1} J^T (V - V_ref)
        JtJ = self.J.T @ self.J
        JtV = self.J.T @ (voltage - self.V_ref)
        reg = self.lambda_reg * np.eye(JtJ.shape[0])
        delta_sigma = np.linalg.solve(JtJ + reg, JtV)
        sigma_0 = self.sigma_ref + delta_sigma
        return sigma_0, info
```

**优势**：
- ✅ 正则化防止过拟合
- ✅ 输出范围更稳定
- ✅ RE预期降至~1.0

#### B. 使用Gauss-Newton迭代

```python
# 迭代优化
for i in range(n_iterations):
    J = compute_jacobian(sigma_current)
    r = voltage - forward_model(sigma_current)
    delta = solve(J^T J + λI, J^T r)
    sigma_current = sigma_current + alpha * delta
```

**优势**：
- ✅ 非线性优化，精度更高
- ✅ RE预期降至~0.5
- ❌ 计算成本较高

#### C. 使用现有pyEIT的GREIT

```python
# pyEIT已实现
from pyeit.eit.recon import GREIT

greit = GREIT(mesh, electrode_config)
sigma_0 = greit.solve(voltage)
```

---

### 🥉 优先级3：减小网格分辨率（内存优化）

**当前问题**：
- n_elems = 11466（mesh_config.yaml: h0=0.0025）
- 与train_config.yaml不一致（n_elems=4424）

**改进方案**：

#### 方案A：统一使用粗网格

```yaml
# config/mesh_config.yaml
mesh_resolution: 0.004  # 从0.0025改为0.004
# 结果：n_elems ≈ 4424（减少60%）
```

**优势**：
- ✅ 内存减少60%
- ✅ 训练速度提升2-3倍
- ✅ 与现有配置一致
- ⚠️ 空间分辨率下降

#### 方案B：保持细网格，但预计算降采样

```python
# 在数据预处理时降采样
sigma_0_coarse = downsample(sigma_0_fine, target_n_elems=4424)
g_coarse = downsample(g_fine, target_n_elems=4424)
```

---

## 🔧 具体实施方案

### 方案1：快速验证（推荐）⭐⭐⭐⭐⭐

**目标**：快速验证ResidualEIT架构的有效性

**步骤**：

1. **使用mixed_dataset**（如果已有预计算特征）
   ```bash
   # 修改配置
   vim config/residual_eit_config.yaml
   # 将 dataset_path 改为 "./data/generated/mixed_dataset.h5"
   
   # 重新训练
   python train_residual_eit.py
   ```

2. **如果mixed_dataset没有预计算特征**
   ```bash
   # 预计算残差特征
   python data/precompute_residual_features.py \
     --h5 data/generated/mixed_dataset.h5 \
     --jacobian data/generated/jacobian.npy \
     --method bp
   
   # 然后训练
   python train_residual_eit.py
   ```

**预期效果**：
- RE: 2.096 → ~0.3-0.5（改善75-85%）
- 训练时间：2-3小时

---

### 方案2：改进传统重建（长期方案）⭐⭐⭐⭐

**目标**：提升σ₀质量，让残差学习更有效

**实施步骤**：

1. **实现Tikhonov重建**
   ```bash
   # 修改 models/traditional/reconstructor.py
   # 添加 TikhonovReconstructor 类
   
   # 测试
   python -c "from models.traditional.reconstructor import TikhonovReconstructor; print('OK')"
   ```

2. **预计算新的残差特征**
   ```bash
   python data/precompute_residual_features.py \
     --h5 data/generated/mixed_dataset.h5 \
     --jacobian data/generated/jacobian.npy \
     --method tikhonov \
     --lambda_reg 0.01
   ```

3. **训练对比**
   ```bash
   # 对比 BP vs Tikhonov
   python train_residual_eit.py --config config/residual_eit_config_bp.yaml
   python train_residual_eit.py --config config/residual_eit_config_tikhonov.yaml
   ```

**预期效果**：
- σ₀ RE: 3.34 → ~1.0
- 最终 RE: 2.096 → ~0.2

---

### 方案3：启用GAT（如果内存充足）⭐⭐⭐

**条件**：
- 减小网格分辨率后（n_elems ≤ 5000）
- 或使用梯度累积（batch_size=4）

**步骤**：

1. **调整配置**
   ```yaml
   # config/residual_eit_config.yaml
   model:
     use_gat: true
     n_heads: 2  # 从4改为2
   
   training:
     batch_size: 4  # 从8改为4
     grad_accum_steps: 2  # 梯度累积
   ```

2. **训练**
   ```bash
   python train_residual_eit.py
   ```

**优势**：
- ✅ GAT可能提升性能5-10%
- ✅ 学习自适应边权重

**风险**：
- ⚠️ 内存可能仍然不足
- ⚠️ 训练速度变慢

---

## 📊 各方案对比总结

### 性能预期

| 方案 | σ₀ RE | 最终 RE | 训练时间 | 实施难度 |
|------|:-----:|:-------:|:--------:|:--------:|
| **当前（BP+eit_dataset）** | 3.34 | 2.096 | 2-3h | - |
| **BP+mixed_dataset** | ~1.5 | ~0.4 | 3-4h | 🟢 简单 |
| **Tikhonov+mixed_dataset** | ~1.0 | ~0.2 | 3-4h | 🟡 中等 |
| **GN+mixed_dataset** | ~0.5 | ~0.15 | 4-5h | 🔴 困难 |
| **GAT（细网格）** | - | OOM | - | ❌ 不可行 |
| **GAT（粗网格）** | - | ~0.18 | 2-3h | 🟡 中等 |

### 推荐路线

```
第1步：使用mixed_dataset（1-2小时）
  ↓ RE: 2.096 → ~0.4
  
第2步：实现Tikhonov重建（半天）
  ↓ RE: ~0.4 → ~0.2
  
第3步：消融实验（1-2天）
  ├─ 对比BP/Tikhonov/GN
  ├─ 对比有监督/弱监督/无监督
  └─ 对比GAT/GNN
  
第4步：优化与部署（1周）
```

---

## 🚀 立即行动

### 当前训练继续观察

当前训练（BP+eit_dataset，use_gat=false）已经启动，建议：

1. **让它跑完**（预计1-2小时完成100 epochs）
2. **观察RE趋势**：是否持续下降
3. **记录最终性能**：作为baseline

### 下一步快速改进

```bash
# 1. 检查mixed_dataset是否可用
ls -lh data/generated/mixed_dataset.h5

# 2. 如果有，修改配置
vim config/residual_eit_config.yaml
# 改 dataset_path: "./data/generated/mixed_dataset.h5"

# 3. 检查是否需要预计算
python3 << 'EOF'
import h5py
with h5py.File('data/generated/mixed_dataset.h5', 'r') as f:
    if 'sigma_0' in f.get('train', {}):
        print("✅ 已有预计算特征，可直接训练")
    else:
        print("❌ 需要预计算残差特征")
EOF

# 4. 如果需要预计算
python data/precompute_residual_features.py \
  --h5 data/generated/mixed_dataset.h5 \
  --jacobian data/generated/jacobian.npy \
  --method bp

# 5. 启动新训练
python train_residual_eit.py
```

---

## 📝 关键结论

### 关于GAT

1. **为什么内存大**：
   - 网格单元多（11466）
   - GAT需要计算所有边的注意力
   - 中间激活占用大量内存

2. **什么时候可以启用GAT**：
   - ✅ 网格单元 ≤ 5000
   - ✅ batch_size ≤ 4
   - ✅ 使用梯度累积
   - ✅ 减小n_heads（4→2）

3. **GAT vs 普通GNN**：
   - 性能差异：预计5-10%
   - 内存差异：70%
   - 建议：先用普通GNN验证架构，最后再试GAT

### 关于数据集

**比算法更重要的是数据质量**！

- ❌ 当前：BP重建RE=3.34（太差）
- ✅ 改进：使用mixed_dataset或Tikhonov
- 🎯 目标：σ₀ RE ≤ 1.0

**核心原则**：
> "Garbage in, garbage out"
> 
> 如果传统重建σ₀很差，神经网络再努力也学不好残差修正。

---

## 🎯 最终建议

**立即实施**：
1. ✅ 保持当前训练运行（观察baseline）
2. 🔴 使用mixed_dataset重新训练（快速提升）
3. 🟡 实现Tikhonov重建（核心改进）

**暂缓实施**：
- ⏸️ GAT（等粗网格验证后再试）
- ⏸️ Gauss-Newton（复杂度高，先用Tikhonov）

**预期效果**：
- 当前RE=2.096 → 目标RE≤0.2
- 改善幅度：90%+
