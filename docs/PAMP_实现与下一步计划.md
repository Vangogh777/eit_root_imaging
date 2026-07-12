# PAMP 模块实现与 ConvSpatialEIT 下一步计划

> 日期: 2026-07-01
> 模型: ConvSpatialEIT v3 (PAMP)
> 状态: PAMP 已实现, 混合数据训练进行中

---

## 一、已完成：PAMP 模块实现

### 1.1 PhysicsAwareMessagePassing 类

**文件**: `models/conv_spatial_eit.py`

将 Jacobian 互灵敏度注入 GNN 消息传递，使消息的方向和强度受物理约束：

| 组件 | 说明 |
|------|------|
| 消息 MLP | `MLP([h_i, h_j, J_mutual, J_ratio])` — 物理特征嵌入消息 |
| 灵敏度门控 | `Sigmoid(MLP([J_mutual, J_ratio]))` — 调制消息强度 |
| 更新 MLP | `MLP([h_i, aggregated_msg])` — 残差更新 |
| 显存优化 | 分块处理边 (chunk_size=5000)，batch=8 仅 ~2.5GB |

### 1.2 物理边特征：_compute_physics_edge_features

| 特征 | 含义 | 归一化 |
|------|------|--------|
| J_mutual | Jacobian 列余弦相似度 — 两个单元对测量的联合影响 | [-1,1] → [0,1] |
| J_ratio | 灵敏度范数比 (log) — 测量对不同位置的敏感度差异 | 截断 [-3,3] → [-1,1] |

### 1.3 训练接口

```bash
python train_conv_spatial.py --use_pamp --data mixed_dataset.h5 --mode supervised --epochs_sup 80
```

---

## 二、当前训练状态

| 项目 | 值 |
|------|-----|
| 模型 | ConvSpatialEIT + PAMP |
| 数据 | mixed_dataset.h5 (4424网格, 20000样本) |
| 参数 | hd256, 4 层 PAMP, 6.8M |
| 进度 | ~epoch 6/80, ~4.4 it/s |
| 当前 RE | ~0.339 (epoch 5, 持续下降中) |
| GPU | RTX 4090, 显存 ~8GB |
| 预计完成 | ~11 小时后 |

---

## 三、下一步详细计划

### Phase 2: 评估与对比（训练完成后，1-2 天）

#### 2.1 PAMP 评估

```bash
# 评估 PAMP 模型在所有测试集上
python evaluate_conv_spatial.py \
  --checkpoint checkpoints/<run_id>/best_supervised.pt \
  --data data/generated/mixed_dataset.h5 \
  --mesh_config config/mesh_config.yaml
```

评估指标: RE, CC, SSIM, PSNR, IoU
测试集:
- `test` — 标准测试 (500 样本)
- `test_low_noise` — 低噪声
- `test_high_noise` — 高噪声
- `test_near_boundary` — 近边界含物
- `test_extrap` — 外推场景

#### 2.2 Baseline 对比（非 PAMP SimpleGNN）

用同样参数（hd256, 4层, 80 epoch, mixed_dataset）训练非 PAMP 版本：

```bash
python train_conv_spatial.py \
  --data data/generated/mixed_dataset.h5 \
  --mesh_config config/mesh_config.yaml \
  --hidden_dim 256 --gnn_layers 4 \
  --mode supervised --epochs_sup 80 \
  --batch_size 8 --grad_accum_steps 2 \
  --mcl_mode full_fem
```

关键对比矩阵:

| 实验 | GNN 类型 | 边特征 | 预期 RE |
|------|---------|--------|---------|
| A1 (PAMP) | PAMP | Jacobian物理 (2维) | ? (训练中) |
| A2 | SimpleGNN | Jacobian物理 (2维) | ? |
| A3 | PAMP | 几何特征 (原有4维) | ? |
| A4 (Baseline) | SimpleGNN | 几何特征 (原有4维) | ~0.108 (历史) |

#### 2.3 消融实验

**架构消融 (A1-A4)**:
- A1: PAMP + Jacobian 物理边特征 (完整模型)
- A2: SimpleGNN + Jacobian 物理边特征 (同特征不同机制)
- A3: PAMP + 几何边特征 (同机制不同特征)
- A4: SimpleGNN + 几何边特征 (当前 baseline)

**组件消融 (B1-B5)**:
- B1: PAMP 完整 (J_mutual + J_ratio + 门控) — baseline
- B2: PAMP 无门控 (去掉 sensitivity_gate)
- B3: PAMP 仅 J_mutual (去掉 J_ratio)
- B4: PAMP 随机边特征 (物理特征替换为噪声)
- B5: SimpleGNN + Jacobian 加权聚合 (无门控)

---

### Phase 3: 11466 精细网格训练（PAMP + 混合数据，3-5 天）

#### 3.1 生成 11466 混合数据集

当前已有 11466 单形状数据:
- `circle_dataset_11466.h5`
- `ellipse_dataset_11466.h5`
- `square_dataset_11466.h5`
- `double_circle_dataset_11466.h5`
- `near_boundary_dataset_11466.h5`

需要生成混合数据集，或修改训练脚本支持多文件混合采样。

```bash
# 方法 A: 生成新的混合数据集（会输出 mixed_dataset.h5 到指定目录）
python data/generate_mixed_dataset.py \
  --config config/mesh_11466_config.yaml \
  --output data/generated/11466 \
  --n_train 20000 --n_val 500 --n_test 500
# 之后重命名: mv data/generated/11466/mixed_dataset.h5 data/generated/mixed_dataset_11466.h5

# 方法 B: 直接用 --data 指定训练结果最好的单个形状数据集作为起点
```

#### 3.2 PAMP hd512 训练 (11466)

```
模型: ConvSpatialEIT + PAMP
参数: hd512, 4 层, ~9.7M
数据: mixed_dataset_11466.h5 (或 5 形状交替)
训练: 80-150 epoch 监督
Baseline: RE=0.073 (ellipse 单形状, hd512, 300ep)
目标: RE < 0.07 (混合数据上超越单形状结果)
```

#### 3.3 训练优化

1. **GAT 变体消融** — 在 `hidden_dim=256` 下快速测试 GAT layer 替代 PAMP 上层
2. **学习率调度** — 尝试 OneCycleLR 替代 CosineAnnealing
3. **数据增强** — 加入 VoltageMasking (20% 随机遮罩)
4. **更长训练** — 参考 ellipse 结果 (300 epoch → RE=0.073), 混合数据可能需更多 epoch

---

### Phase 4: 论文级优化（1-2 周）

#### 4.1 通用性验证

| 实验 | 训练数据 | 测试数据 | 验证目标 |
|------|---------|---------|---------|
| D1 | mixed 5 shapes | 各形状分别测试 | 通用模型泛化性 |
| D2 | 单形状 | 同形状测试 | 专用模型上界 |
| D3 | D1 vs D2 差距 | — | 泛化 gap 量化 |

#### 4.2 鲁棒性实验

| 实验 | 条件 | 验证目标 |
|------|------|---------|
| E1 | 测试噪声 SNR=40,30,20,10 dB | 噪声鲁棒性 |
| E2 | 电极偏移 ±1,2,5mm | 模型误差鲁棒性 |

#### 4.3 传统方法对比

对比 GN (Gauss-Newton)、GREIT 在相同测试集上的表现：

```python
from models.traditional.reconstructor import EITReconstructor
recon = EITReconstructor(solver, method='gn')
sigma_gn = recon.solve(voltages)
```

#### 4.4 与其他 SOTA 论文定量对比

| 方法 | RE | 条件 | 备注 |
|------|-----|------|------|
| GN/GREIT (传统) | ? | 同测试集 | 必须对齐 |
| GraphEIT (2024) | 文献值 | 16电极, 2D | 不一定相同配置 |
| CNN+PINN (2025) | 文献值 | — | 参考对比 |
| **我们的 PAMP** | **?** | **混合数据** | **目标 RE<0.07** |

---

### Phase 5: 无监督训练修复（长期目标）

当前问题: 监督 (RE=0.108) → 无监督微调后退化到 RE=0.538

修复方向:

1. **全 FEM 覆盖** — 每步对所有样本做完整 FEM 正解（增加 fem_interval 频率）
2. **混合损失权重** — `L_total = 0.95*L_sup + 0.05*L_phys`（极小物理权重做软约束）
3. **PAMP 物理一致性** — 利用 PAMP 的 Jacobian 特征做自洽约束
4. **迭代优化** — 参考 GN 的迭代策略，用 FEM 梯度替代 Jacobian 近似
5. **对比学习预训练** — 先用对比学习替代监督预训练

---

## 四、代码修改清单

### 已修改

| 文件 | 改动 | 状态 |
|------|------|------|
| `models/conv_spatial_eit.py` | 新增 `PhysicsAwareMessagePassing` 类 + `_compute_physics_edge_features` + `use_pamp` 参数 + forward 条件路径 | ✅ 已完成 |
| `train_conv_spatial.py` | 新增 `--use_pamp` 参数 + 自动 Jacobian 加载 + meta 记录 | ✅ 已完成 |

### 待修改

| 文件 | 改动 | 优先级 |
|------|------|--------|
| `evaluate_conv_spatial.py` | 支持 PAMP 模型加载和物理边特征恢复 | P1 |
| `evaluate_current_run.py` | 适配 PAMP meta 自动检测 | P1 |
| `config/train_config.yaml` | 新增 `model.use_pamp` 配置项 | P2 |
| `config/train_config_11466.yaml` | 同上 | P2 |

---

## 五、时间线估计

```
Phase 1: PAMP 实现              ✅ 已完成 (2026-07-01)
Phase 2: 评估与对比             🔄 训练中 (1-2 天)
Phase 3: 11466 精细网格训练     📅 ~3-5 天
Phase 4: 论文级优化             📅 ~1-2 周
Phase 5: 无监督修复 (可选)      📅 长期
```

**里程碑**:
- 7/2 — PAMP 混合数据 80 epoch 完成, 基线对比结果出炉
- 7/4 — 11466 混合数据 PAMP hd512 训练启动
- 7/7 — 完成消融实验, 确定论文基线
- 7/14 — 鲁棒性实验 + 传统方法对比完成
