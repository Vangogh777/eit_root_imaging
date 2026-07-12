# EIT 植物根部成像系统现状分析

> 最后更新: 2026-06-23

---

## 📋 目录

1. [项目概述](#项目概述)
2. [系统架构](#系统架构)
3. [训练现状](#训练现状)
4. [性能瓶颈](#性能瓶颈)
5. [已知问题](#已知问题)
6. [改进方向](#改进方向)

---

## 项目概述

### 基本信息

**项目目标**：桶式（bucket-type）2D EIT 系统，用于植物根部的无监督成像

**硬件场景**：
- 圆柱形桶（直径 ≈ 20cm）
- 单环 16 电极
- 2D 截面采集
- 多频测量（6个频率：1kHz - 500kHz）

**技术栈**：
- Python + PyTorch
- pyEIT（FEM 正演求解器）
- HDF5 数据存储
- TensorBoard + wandb 日志
- ONNX 部署导出

### 核心创新点

**无监督训练哲学**：
- 不使用真实 σ（电导率分布）作为监督信号
- 仅依赖物理约束进行训练
- 损失函数：`L_total = λ_m * L_meas + λ_tv * L_tv + λ_freq * L_freq + λ_blc * L_blc + λ_smooth * L_smooth`

---

## 系统架构

### 模型家族

| 模型 | 文件 | 状态 | 说明 |
|------|------|------|------|
| **ConvSpatialEIT** | `models/conv_spatial_eit.py` | ✅ 主要使用 | 卷积空间模型，两阶段训练 |
| **SF-SBLC** | `models/sf_sblc.py` | ✅ 已实现 | 空间-频率共享 + 基础层校正 |
| **PhysicsInformedEIT** | `models/universal_eit.py` | ⚠️ 实验性 | 物理信息约束模型 |
| **TwoStageEITModel** | `models/two_stage_model.py` | ⚠️ 实验性 | 传统反演 + 神经优化 |
| **EITModelGNN** | `models/eit_gnn_model.py` | ⚠️ 实验性 | 图神经网络 |
| **PhysicsGNN** | `models/physics_gnn.py` | ⚠️ 实验性 | 物理约束 GNN |

### ConvSpatialEIT 架构（当前主模型）

```
输入: 边界电压 (B, n_freq=6, n_meas=208)
  ↓
特征编码器
  ↓
空间重建网络（基于图卷积）
  ↓
输出: 电导率分布 (B, n_elems=4424)
```

**关键特性**：
- 两阶段训练：有监督预训练（快速收敛）→ 无监督精调（物理约束）
- 支持 `jacobian` / `full_fem` / `hybrid` 三种物理约束模式
- 图结构建模 FEM 网格的空间关系

### SF-SBLC 架构（原始主模型）

```
输入: 边界电压 (B, n_freq=6, n_meas=208)
  ↓
SharedEncoder → 多频共享编码
  ↓
BaseLayerCorrection (BLC) → 系统伪影抑制
  ↓
FrequencyFusionDecoder → 频率融合解码
  ↓
ResNetBackbone → 深度残差重建
  ↓
输出: 电导率分布 (B, n_elems=4424)
```

**输出包含可解释中间量**：
- `base_map`: 基础层估计
- `freq_weights`: 频率注意力权重
- `blc_gates`: BLC 校正门控值

---

## 训练现状

### 已完成训练

#### 最佳性能模型

**模型路径**：`checkpoints/20260622_015538_v2_both_hd256/best.pt`

**性能指标**：
```
RE (相对误差)  = 0.1928 ± 0.0932
CC (相关系数)  = 0.9546 ± 0.0443
推理速度       = 1.44 ms/样本
参数量         = 6.1M
```

**训练配置**：
- 训练模式：两阶段（有监督预训练 + 无监督精调）
- 模型架构：ConvSpatialEIT, hidden_dim=256
- 数据集：`mixed_dataset.h5` (112MB)
- 评估数据集：test split (500 samples)

#### 历史最佳性能

**日期**：2026-06-17
**配置**：hidden_dim=512
**性能**：RE=0.103

**⚠️ 性能退化分析**：
- hd256 vs hd512：RE 从 0.103 → 0.193（退化 87%）
- 可能原因：
  1. 模型容量不足（hd256 参数量减半）
  2. 训练策略差异（超参数、损失权重）
  3. 数据集变化

#### 最新训练

**模型路径**：`checkpoints/20260622_231207_v2_unsupervised_hd256/final.pt`

**训练详情**：
- 训练轮数：50 epochs（已完成）
- 物理约束模式：`full_fem`（修复雅可比线性近似问题）
- 损失：最终 Loss = 0.3696
- 梯度警告：频繁出现大梯度（最大 980.70）

### 正在运行的训练

**当前状态**：3 个训练任务并行运行

| Run ID | 模型 | 状态 | 开始时间 |
|--------|------|------|----------|
| `20260623_174814` | v2_both_hd256 | running | 2026-06-23 17:48 |
| `20260623_174548` | v2_both_hd256 | running | 2026-06-23 17:45 |
| `20260623_174402` | v2_both_hd256 | running | 2026-06-23 17:44 |

### 数据资源

| 数据集 | 大小 | 说明 |
|--------|------|------|
| `mixed_dataset.h5` | 112MB | 混合数据集（主用） |
| `circle_dataset.h5` | 110MB | 圆形域数据 |
| `eit_dataset.h5` | 1.6MB | 原始数据集 |
| `jacobian.npy` | 55MB | 预计算雅可比矩阵 |

**数据生成流程**：
1. `RootSystemGenerator` → 随机根系结构（直根/须根/鲱骨型）
2. `EITForwardSolver` → pyEIT FEM 正演
3. 存储为 HDF5 格式（电压测量 + 真实 σ）

---

## 性能瓶颈

### 1. 重建精度瓶颈

**当前性能**：
- RE = 0.193（目标：≤ 0.15）
- CC = 0.955（目标：≥ 0.97）

**瓶颈分析**：

#### A. 物理约束不足

**问题**：雅可比线性近似在 σ 偏离参考点后失效

```
雅可比模式: RE=0.648, CC=-0.01 (已失效)
FEM模式:    RE=0.193, CC=0.955  (有效但仍不够)
```

**根本原因**：
- 雅可比矩阵仅在 σ_ref=0.01 S/m 附近成立
- 真实根系 σ=0.05 S/m，偏离线性化点 5 倍
- 导致测量一致性损失无法正确引导模型

**已采取的措施**：
- 切换到 `full_fem` 模式（训练时间增加 3-5x）
- 但 FEM 模式本身也有近似误差（梯度通过 Jacobian 回传）

#### B. 模型容量问题

**问题**：hidden_dim=256 可能不足以表达复杂的根系结构

**证据**：
- 历史最佳：hd512, RE=0.103
- 当前模型：hd256, RE=0.193
- 容量差距：6.1M vs ~12M 参数

#### C. 无监督训练的不适定性

**问题**：EIT 逆问题本身是不适定的，无监督训练更难收敛

**表现**：
- 训练损失下降，但验证指标波动
- 大梯度频繁出现（说明优化困难）
- 需要强正则化约束

### 2. 训练效率瓶颈

**问题**：完整 FEM 模式训练缓慢

**时间估算**：
- 单个 epoch: ~15 分钟（full_fem 模式）
- 单次训练: ~12.5 小时（50 epochs）
- 总训练成本高（多次实验迭代）

**GPU 利用率**：
- 训练时：90%+
- 但 FEM 求解是 CPU bound（Python + numpy）

### 3. 泛化能力瓶颈

**潜在问题**：模型可能在仿真数据上过拟合

**证据**：
- 仅在仿真数据上评估
- 真实测量数据可能存在：
  - 电极接触阻抗
  - 测量噪声
  - 系统漂移
  - 3D 效应

---

## 已知问题

### 1. 评估脚本兼容性问题

**问题**：`evaluation/evaluate.py` 无法加载新 checkpoint

```python
KeyError: 'model_state_dict'
```

**原因**：
- 新训练脚本使用不同的 checkpoint 格式
- 旧格式：`{'model_state_dict': ...}`
- 新格式：`{'model': ...}` 或直接 state_dict

**影响**：无法使用标准评估流程

**临时方案**：使用 `evaluate_conv_spatial.py` 等专用脚本

**修复建议**：
```python
def extract_model_state(ckpt):
    """兼容多种 checkpoint 格式"""
    if isinstance(ckpt, dict):
        for key in ("model", "model_state_dict"):
            if key in ckpt and isinstance(ckpt[key], dict):
                return ckpt[key]
    return ckpt
```

### 2. 训练过程大梯度警告

**问题**：训练过程中频繁出现梯度爆炸

```
⚠ 大梯度: 980.70 | loss=0.5746
```

**已采取措施**：
- 梯度裁剪：`grad_clip: 1.0`
- 但警告仍然频繁出现

**可能原因**：
1. 损失函数权重不平衡
2. 物理约束与重建目标冲突
3. 网络结构设计问题

**建议**：
- 调整损失权重（特别是 λ_m 和 λ_tv）
- 检查物理约束损失的计算逻辑
- 考虑增加梯度裁剪强度或使用自适应裁剪

### 3. 网格分辨率不一致

**问题**：配置文件中网格参数存在矛盾

| 配置文件 | 参数 | 单元数 |
|---------|------|--------|
| `mesh_config.yaml` | `mesh_resolution: 0.0025` | ~11466 |
| `train_config.yaml` | `n_elems: 4424` | 4424 |

**影响**：
- 容易导致维度不匹配错误
- 影响雅可比矩阵和模型输出的一致性

**修复建议**：
- 统一使用同一配置源
- 或在训练脚本中自动从 mesh_config 推导 n_elems

### 4. 无监督训练稳定性

**问题**：无监督训练的收敛不稳定

**表现**：
- 验证指标震荡
- 不同随机种子结果差异大
- 需要多次运行取最优

**可能原因**：
- 损失函数景观复杂
- 多个局部最优
- 物理约束与数据拟合的权衡

### 5. 缺少真实数据验证

**问题**：所有评估都在仿真数据上进行

**风险**：
- 模型可能无法处理真实测量噪声
- 电极建模误差
- 环境因素影响

**建议**：
- 收集真实测量数据
- 进行仿真-to-真实（sim-to-real）迁移实验
- 或使用域自适应技术

---

## 改进方向

### 短期改进（1-2 周）

#### 1. 修复评估和监控工具

**优先级**：🔴 高

**任务**：
- [ ] 修复 `evaluation/evaluate.py` 的 checkpoint 兼容性
- [ ] 建立自动化评估流程
- [ ] 完善训练监控（实时指标、早停）

#### 2. 分析性能退化原因

**优先级**：🔴 高

**任务**：
- [ ] 对比 hd256 vs hd512 的训练曲线
- [ ] 检查超参数差异（学习率、损失权重）
- [ ] 确认数据集是否一致
- [ ] A/B 测试验证

#### 3. 优化训练效率

**优先级**：🟡 中

**任务**：
- [ ] 实现 FEM 求解的 GPU 加速
- [ ] 或使用 Jacobian + 定期 FEM 校正的混合策略
- [ ] 优化数据加载（内存映射、预取）

### 中期改进（1-2 月）

#### 4. 增强物理约束

**优先级**：🔴 高

**方向**：
- **改进测量一致性损失**：
  - 自适应 Jacobian 更新（在线重新计算）
  - 使用更高阶的 Taylor 展开
  - 神经网络学习正向模型

- **增加物理先验**：
  - 电导率范围约束（土壤 vs 根系）
  - 空间连续性约束
  - 时间一致性（如果有时序数据）

#### 5. 改进模型架构

**优先级**：🟡 中

**方向**：
- **注意力机制**：
  - 跨频率注意力（已有，但可增强）
  - 空间注意力（关注根系区域）

- **多尺度重建**：
  - 粗到细的重建策略
  - 金字塔式特征提取

- **图神经网络**：
  - 更好地利用 FEM 网格拓扑
  - 边特征建模（电极连接关系）

#### 6. 混合监督策略

**优先级**：🟡 中

**思路**：
- 使用少量标注数据 + 大量无标注数据
- 半监督学习（pseudo-labeling）
- 自监督预训练（掩码电压重建）

### 长期改进（3-6 月）

#### 7. 真实数据验证

**优先级**：🔴 高

**任务**：
- [ ] 建立真实数据采集流程
- [ ] 标注真实根系分布（CT/MRI 对比）
- [ ] 研究 sim-to-real 迁移方法
- [ ] 建立真实数据测试集

#### 8. 3D 成像扩展

**优先级**：🟢 低（取决于硬件）

**方向**：
- 多层电极环 → 3D 重建
- 3D FEM 网格
- 计算效率优化（3D FEM 更慢）

#### 9. 部署优化

**优先级**：🟡 中

**任务**：
- [ ] ONNX 模型优化（量化、剪枝）
- [ ] 边缘设备部署（Jetson、树莓派）
- [ ] 实时成像系统

---

## 性能目标与路线图

### 当前性能

| 指标 | 当前值 | 目标值 | 差距 |
|------|--------|--------|------|
| RE | 0.193 | ≤ 0.15 | -23% |
| CC | 0.955 | ≥ 0.97 | -1.5% |
| 推理速度 | 1.44 ms | ≤ 10 ms | ✅ 达标 |

### 里程碑

#### Milestone 1: 稳定性（预计 2 周）
- [ ] 修复所有已知工具问题
- [ ] 建立自动化训练-评估流程
- [ ] 达到稳定的 RE ≤ 0.18

#### Milestone 2: 精度提升（预计 1.5 月）
- [ ] 实现改进的物理约束
- [ ] 优化模型架构
- [ ] 达到 RE ≤ 0.15, CC ≥ 0.97

#### Milestone 3: 真实数据验证（预计 3 月）
- [ ] 收集真实测量数据
- [ ] 验证 sim-to-real 性能
- [ ] 优化鲁棒性

---

## 关键文件索引

### 训练脚本
- `train.py` - 标准 SFSBLC 训练
- `train_conv_spatial.py` - 两阶段训练（当前主用）
- `train_server.py` - GPU 服务器训练
- `train_m1.py` - M1 Mac 优化训练

### 评估脚本
- `evaluation/evaluate.py` - 标准评估（需修复）
- `evaluation/validate.py` - 验证脚本
- `evaluate_conv_spatial.py` - ConvSpatialEIT 专用评估
- `evaluate_current_run.py` - 当前训练评估

### 配置文件
- `config/mesh_config.yaml` - 网格配置
- `config/train_config.yaml` - 训练配置
- `config/pyeidors_train_config.yaml` - PyEIDORS 配置

### 核心模块
- `models/conv_spatial_eit.py` - 主模型
- `training/loss.py` - 损失函数
- `data/eit_forward.py` - FEM 正演
- `data/root_simulator.py` - 根系生成

### 工具脚本
- `serve_results.py` - 结果展示服务器
- `monitor_training.py` - 训练监控
- `visualize_results.py` - 结果可视化

---

## 参考文献

### EIT 基础
- Holder, D. S. (2005). Electrical Impedance Tomography: Methods, History and Applications
- pyEIT 文档: https://github.com/liubenyuan/pyEIT

### 深度学习 + EIT
- Hamilton, S. J., & Hauptmann, A. (2018). Deep D-Bar for Electrical Impedance Tomography
- Seo, J. K., & Woo, E. J. (2013). Nonlinear Inverse Problems in EIT

### 相关工作
- Adler, A., & Guardo, R. (1996). A neural network approach to electrical impedance tomography
- Martin, S., & Choi, C. T. (2016). A post-processing method for three-dimensional EIT imaging

---

## 联系与资源

- **项目路径**: `/home/ubuntu/EIT/eit_root_imaging`
- **结果服务器**: `http://localhost:8080` (serve_results.py)
- **训练日志**: `train_full_fem.log`, `train.log`
- **检查点目录**: `checkpoints/`
- **训练记录**: `training_records/`

---

**文档维护**：
- 定期更新训练状态
- 记录性能改进
- 追踪问题和解决方案
