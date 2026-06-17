# EIT 深度学习前沿路线综述 (2023-2026)

> 2026-06-17 | 基于 PubMed / IEEE Xplore / Neural Networks 最新论文
> 基准: ConvSpatialEIT v2, gnn_hidden=512, 4.2M params, RE=0.108

---

## 核心论文总览

| 代号 | 论文全称 | 期刊 | 时间 | PMID/DOI |
|------|----------|------|:---:|------|
| **GSR** | Gaussian Splatting-Based Reconstruction for EIT | IEEE TMI | 2026.5 | 10.1109/TMI.2025.3647129 |
| **SDEIT** | Semantic-Driven Electrical Impedance Tomography | Neural Networks | 2026.5 | 10.1016/j.neunet.2025.108492 |
| **DeepPrior** | Deep Prior Embedding Method for EIT | Neural Networks | 2025.8 | 10.1016/j.neunet.2025.107419 |
| **BoundaryAttn** | Post-Processing for EIT Integrating Boundary Attention | Sensors | 2026.5 | 10.3390/s26103117 |
| **StructEIT** | Realistic 3D EIT Model Generation from CT Scans | Physiol Meas | 2026.4 | 10.1088/1361-6579/ae5587 |
| **MMV-Net** | Multi-Frequency EIT via ADMM Unfolding + Self-Attention | IEEE TNNLS | 2023 | PMID:35263263 |
| **RFNetEIT** | 3D EIT via Feature Map Reconfiguration + Residual Network | PeerJ | 2024 | PMC11042020 |

---

## 演进谱系（同一中国团队主导）

```
2023 ── MMV-Net (ADMM 展开 + 自注意力, IEEE TNNLS)
         └── 多频相关性显式建模，优化框架展开

2025 ── DeepPrior (深度先验嵌入, Neural Networks)
         └── 将 EIT 物理先验嵌入神经网络训练

2026 ── SDEIT (语义驱动 EIT, Neural Networks)
         └── 语义分割先验注入正则化

2026 ── GSR (Gaussian Splatting EIT, IEEE TMI)
         └── 3D 高斯泼溅用于 EIT 体积重建
```

**核心作者**: Liu D, Deng J, Wang J, Wu Y（同一研究组，连续 3 篇顶刊）

---

## 各方法详细对比

### GSR: Gaussian Splatting EIT (2026, IEEE TMI)

**核心思想**: 将 3DGS (3D Gaussian Splatting) 从图形学引入 EIT

```
传统: voltages → 2D pixel/2D mesh → sigma
GSR:  voltages → 3D Gaussians (position, covariance, opacity) → 体积重建
```

**架构**:
1. 边界电压编码为 latent code
2. Latent code 解码为一组 3D Gaussians（位置 μ, 协方差 Σ, 透明度 α, 颜色 c）
3. 可微分渲染 → 任意视角的 2D 投影
4. Physics loss: 渲染投影的 FEM 正解与测量电压一致

**优势**:
- 天然支持 3D 体积重建（非 2D 截面）
- 显式几何表示（不依赖网格分辨率）
- 可微分渲染管道端到端训练
- 任意视角可视化

**劣势**:
- 需要 3D 训练数据（难获取）
- 训练时间长（Gaussian 优化本质上是迭代的）
- 工程复杂度高

**与我们的差距**: 4 级（3D 表示学习 vs 2D GNN，差 2 代）

---

### SDEIT: Semantic-Driven EIT (2026, Neural Networks)

**核心思想**: 用语义分割先验约束 EIT 重建

```
voltages → 重建网络 → sigma_raw
                      ↓
semantic_prior (解剖结构掩膜) → 正则化 → sigma_refined
```

**架构**:
1. 重建网络产生粗 σ 图
2. 语义分割网络产生器官/组织掩膜
3. 语义先验损失: 同语义区域内 σ 应一致，不同区域边界允许跳变
4. 联合训练: reconstruction loss + semantic consistency loss

**优势**:
- 解剖先验强约束 → 抑制伪影
- 多器官场景适用（肺、心脏、肝脏）
- 不需要 paired CT/MRI 标注（语义掩膜可从公开模型获取）

**劣势**:
- 依赖语义分割模型的准确性
- 不适用于无先验知识的场景（如植物根系）

**与我们的差距**: 3 级（需要语义分支，但 GNN 保留）

---

### DeepPrior: Deep Prior Embedding (2025, Neural Networks)

**核心思想**: 将 EIT 物理先验（灵敏度、正向算子）显式嵌入网络

```
voltages ────→ Reconstruction Network
                  ↑
Jacobian_matrix ──┘ (物理先验条件)
```

**架构**:
1. 预计算 Jacobian 敏感度矩阵 J
2. J 作为条件输入与电压特征融合
3. 训练时: loss 同时包含 reconstruction MSE 和 physics consistency

**优势**:
- 利用 EIT 的物理约束（非黑盒学习）
- Jacobian 信息告诉网络"哪里敏感、哪里不敏感"
- 改动小，可叠加到现有架构

**劣势**:
- Jacobian 依赖参考 σ（非自适应）
- 仍依赖网格离散化

**与我们的差距**: 2 级（直接可叠加到 ConvSpatialEIT 上）

---

### BoundaryAttn: 边界注意力后处理 (2026, Sensors)

**核心思想**: 用注意力机制聚焦 EIT 的边界区域（敏感区）

```
sigma_coarse → BoundaryAttention(edge_features) → sigma_refined
```

- 专门针对 EIT 中"中心灵敏度低、边界灵敏度高"的特点
- 后处理模块：不改变主干网络

**与我们的差距**: 2 级（对应 SDD Phase 2.2 GAT 注意力）

---

### StructEIT: CT→EIT 3D 模型生成 (2026, Physiol Meas)

**核心思想**: 从真实 CT 扫描自动生成 3D EIT 训练数据

```
CT_scans → 组织分割 → 电导率赋值 → 3D FEM mesh → EIT 前向仿真
```

- 解决了 EIT 训练数据稀缺的问题
- 生成的模型具有真实解剖结构

**与我们的差距**: 无关（这是数据生成工具，不是重建算法）

---

## 各方法能力矩阵

| 能力 | ConvSpatial (我们) | MMV-Net | DeepPrior | SDEIT | GSR | BoundaryAttn |
|------|:---:|:---:|:---:|:---:|:---:|:---:|
| 2D 重建 | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| 3D 重建 | ❌ | ❌ | ❌ | ❌ | ✅ | ❌ |
| 多频利用 | ⚠️刚加 | ✅ | ❌ | ❌ | ❌ | ❌ |
| 物理先验 | ⚠️可选 | ✅ | ✅ | ✅ | ✅ | ❌ |
| 语义先验 | ❌ | ❌ | ❌ | ✅ | ❌ | ❌ |
| 注意力机制 | ❌ | ✅ | ❌ | ❌ | ⚠️隐式 | ✅ |
| 无监督训练 | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ |
| 训练管线 | ✅完整 | ❌无公开 | ❌ | ❌ | ❌ | ❌ |
| 工程成熟度 | ⭐⭐⭐ | ⭐ | ⭐ | ⭐ | ⭐ | ⭐⭐ |
| 发表期刊 | — | IEEE TNNLS | Neural Netw | Neural Netw | IEEE TMI | Sensors |

---

## 推荐演进路线

```
当前 (2026.6)
├── ConvSpatialEIT v2 (RE=0.108)
├── Phase 1 (进行中): 6频融合 + 半监督
│    目标 RE: 0.07-0.08
│
Phase 2 (下阶段)
├── 2a: 边界注意力 GNN (BoundaryAttn 方向)
│    → SimpleGNN → GATLayer (SDD 2.2)
├── 2b: 深度先验注入 (DeepPrior 方向)
│    → Jacobian 通道条件输入
│
Phase 3 (中期)
├── 语义先验注入 (SDEIT 方向)
│    → 解剖/形态先验约束重建
├── 3D 扩展 (StructEIT + GSR 方向)
│    → 从 2D 截面 → 3D 体积
│
Phase 4 (远期)
└── 3D Gaussian Splatting EIT (GSR 方向)
     → 端到端 3D 可微分渲染 EIT
```

### 最小成本最大收益路径（叠加式）

每步都在上一步基础上叠加，不推翻重来：

```
Step 1 (本次): 6频融合 (DeepPrior 方向, 物理先验注入)
    +0 params, RE 0.108→0.07

Step 2 (下次): 边界注意力 GNN (BoundaryAttn + MMV-Net 方向)
    +200K params, RE 0.07→0.05

Step 3: Jacobian 条件输入 (DeepPrior 核心)
    +3 params (1个输入通道), RE 0.05→0.04

Step 4: 语义锚点损失 (SDEIT 方向)
    不改变架构, 损失层面约束

Step 5: 3D 体积扩展 (StructEIT + GSR)
    重大架构变更
```

### 核心判断

1. **ConvSpatialEIT 没有过时** — 它是 2023 水平的合理设计，且在 2025-26 的顶刊论文中，GNN+CNN 混合架构仍然是主流骨架
2. **当前最值得追的方向是 DeepPrior（物理先验注入）** — 最小改动，最大杠杆
3. **GAT 注意力被 2026 论文验证** — BoundaryAttn 就是 attention for EIT，和我们 SDD Phase 2.2 一致
4. **GSR (3D Gaussian Splatting) 是终极方向** — 但需要大量工程投入，建议作为 Phase 4 远期目标
5. **扩散模型在 EIT 中没有真实发表** — 之前在 analysis 文档中的扩散相关内容为推测，不成立

---

## 参考文献

| 编号 | 引用 |
|:---:|------|
| [1] | Liu D et al. "GSR: A Gaussian Splatting-Based Reconstruction Framework for EIT." IEEE TMI, 2026. DOI: 10.1109/TMI.2025.3647129 |
| [2] | Liu D et al. "SDEIT: Semantic-Driven Electrical Impedance Tomography." Neural Networks, 2026. DOI: 10.1016/j.neunet.2025.108492 |
| [3] | Wang J et al. "Deep Prior Embedding Method for Electrical Impedance Tomography." Neural Networks, 2025. DOI: 10.1016/j.neunet.2025.107419 |
| [4] | Zhang L, Wang W. "Post-Processing Algorithm for Leg EIT Integrating Boundary Attention Mechanism." Sensors, 2026. DOI: 10.3390/s26103117 |
| [5] | Jiang Z et al. "StructEIT: Realistic 3D EIT Model Generation from CT Scans." Physiol Meas, 2026. DOI: 10.1088/1361-6579/ae5587 |
| [6] | Chen et al. "MMV-Net: Multi-Frequency EIT via ADMM." IEEE TNNLS, 2023. PMID: 35263263 |
| [7] | Zheng et al. "RFNetEIT: 3D EIT via Feature Map Reconfiguration." PeerJ, 2024. PMC11042020 |
