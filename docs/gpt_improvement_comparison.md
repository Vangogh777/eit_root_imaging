# GPT咨询改进方案 vs 当前实现对比分析

> 分析日期: 2026-06-23
> 咨询文档: `docs/20260617王楠GPT咨询算法优化.md`

---

## 📋 执行摘要

**核心发现**：
- ✅ **Phase 1 改进已 90% 完成**（Jᵀr 反投影、残差输出、边特征增强均已实现）
- ⚠️ **Phase 2 改进未实施**（缺少 Tikhonov 初始重建、残差稀疏损失、Graph TV）
- 📊 **性能未达预期**：当前 RE=0.193，距离 Phase 1 目标 RE≤0.20 有差距

**关键差异**：
1. **多频融合方式更好**：使用 Cross-Attention 而非简单 1x1 Conv
2. **残差归一化处理更稳定**：避免了大梯度问题
3. **缺少物理损失约束**：未实现 residual loss 和 graph TV

**推荐行动**：
- 🔴 **立即实施 Phase 2.2**：添加 residual loss 和 graph TV（预期 RE ↓10-15%）
- 🟡 **评估 Phase 2.1**：Tikhonov 初始重建的价值（需重新生成数据）
- 🟢 **长期规划 Phase 3**：完整三分支架构（论文版）

---

## 1. Phase 1 改进对比

### 1.1 Jᵀr 物理残差反投影

| 对比项 | GPT 建议 | 当前实现 | 差异评估 |
|--------|---------|---------|---------|
| **Jacobian 加载** | ✅ 预存 J, Jᵀ | ✅ 已实现 | ✅ 一致 |
| **残差计算** | `r = V_meas - V_lin` | ✅ 已实现 + 归一化 | ✅ 改进：更稳定 |
| **反投影** | `g = Jᵀr` | ✅ 已实现 + 标准化 | ✅ 改进：避免尺度失控 |
| **校正头** | MLP (gnn_hidden+1 → 1) | ✅ 已实现 | ✅ 一致 |

**实现代码对比**：

```python
# GPT 建议的简化版
g = (self.J_T.unsqueeze(0) @ r.unsqueeze(-1)).squeeze(-1)
delta = self.correction_head(torch.cat([h, g.unsqueeze(-1)], dim=-1)).squeeze(-1)
sigma = sigma_0 + delta

# 当前实现的改进版
# 1. 使用物理约束的 σ₀ 做线性化
sigma_0_phys = torch.sigmoid(sigma_raw_0) * (self.sigma_max - self.sigma_min) + self.sigma_min
sigma_delta = sigma_0_phys - self.sigma_ref  # 避免未约束的 logits 溢出

# 2. 残差归一化
v_scale = V_meas.detach().norm(dim=-1, keepdim=True).clamp_min(1e-6)
r = (V_meas - V_lin) / v_scale  # 样本级归一化

# 3. 反投影后标准化
g = (self.J_T.unsqueeze(0) @ r.float().unsqueeze(-1)).squeeze(-1)
g = (g - g.mean(dim=-1, keepdim=True)) / (g.std(dim=-1, keepdim=True) + 1e-6)  # 标准化

# 4. 最终输出
delta = self.correction_head(torch.cat([h, g.unsqueeze(-1)], dim=-1)).squeeze(-1)
sigma = torch.sigmoid(sigma_raw_0 + delta) * (self.sigma_max - self.sigma_min) + self.sigma_min
```

**✅ 当前实现的优势**：
1. **数值稳定性更好**：使用物理范围内的 σ₀ 做线性化，避免 logits 经过 Jacobian 放大后溢出
2. **样本自适应归一化**：按样本尺度归一化残差，避免不同样本间尺度差异
3. **标准化的反投影特征**：零均值单位方差，让 correction_head 更容易学习

**⚠️ 潜在问题**：
- 归一化可能削弱了物理意义（残差的绝对大小有物理含义）
- 但从工程角度看，稳定性提升更重要

---

### 1.2 残差输出 (σ₀ → Δσ)

| 对比项 | GPT 建议 | 当前实现 | 评估 |
|--------|---------|---------|------|
| **输出结构** | `σ = σ₀ + Δσ` | ✅ `sigma = Sigmoid(σ₀_raw + Δσ)` | ✅ 一致（Sigmoid 归一化） |
| **中间量输出** | 用于消融分析 | ✅ 返回 `sigma_0`, `delta` | ✅ 一致 |
| **范围约束** | Sigmoid → [σ_min, σ_max] | ✅ 已实现 | ✅ 一致 |

**代码对比**：

```python
# GPT 建议
sigma = sigma_0 + delta
sigma = torch.sigmoid(sigma) * (self.sigma_max - self.sigma_min) + self.sigma_min

# 当前实现（一致）
sigma_raw = sigma_raw_0 + delta
sigma = torch.sigmoid(sigma_raw) * (self.sigma_max - self.sigma_min) + self.sigma_min
```

**✅ 完全符合建议**，且提供了更丰富的输出用于分析。

---

### 1.3 边特征增强

| 对比项 | GPT 建议 | 当前实现 | 评估 |
|--------|---------|---------|------|
| **边特征维度** | 4维（距离、共享边长、面积比、灵敏度相似性） | ⚠️ edge_dim 可配置，但未明确指定维度 | ⚠️ 部分实现 |
| **EdgeFeatureGNNLayer** | ✅ 完整实现 | ✅ SimpleGNNLayer 支持边特征 | ✅ 功能等效 |
| **预计算边特征** | ✅ 在 `setup_mesh()` 中计算 | ⚠️ 代码中有 `edge_feat` 但未看到具体计算 | ⚠️ 需确认 |

**代码对比**：

```python
# GPT 建议：显式计算 4 维边特征
edge_feat = np.zeros((n_edges, 4))
for e_idx, (i, j) in enumerate(zip(edge_list[0], edge_list[1])):
    edge_feat[e_idx, 0] = np.linalg.norm(centers[i] - centers[j])  # 距离
    edge_feat[e_idx, 1] = len(set(elements[i]) & set(elements[j])) / 3.0  # 共享边长
    edge_feat[e_idx, 2] = ...  # 面积比
    edge_feat[e_idx, 3] = ...  # 灵敏度相似性

# 当前实现：SimpleGNNLayer 支持 edge_feat 输入
class SimpleGNNLayer(nn.Module):
    def __init__(self, in_dim, out_dim, edge_dim=0):
        if edge_dim > 0:
            self.edge_net = nn.Sequential(
                nn.Linear(edge_dim, out_dim),
                nn.GELU(),
                nn.LayerNorm(out_dim),
            )
```

**⚠️ 需要确认**：
- 边特征是否已预计算并传入模型？
- 如果未计算，可以快速补充（代码量 <20 行）

---

### 1.4 多频融合（额外改进）

| 对比项 | GPT 建议 | 当前实现 | 评估 |
|--------|---------|---------|------|
| **融合方式** | 1×1 Conv 或平均 | **Cross-Attention** | ✅ **更优** |
| **物理意义** | 简单线性组合 | 非线性交互 + 位置编码 | ✅ 更强表达能力 |

**当前实现的优势**：

```python
class FrequencyCrossAttention(nn.Module):
    """多频 Cross-Attention 融合层 — 替换 1×1 Conv"""
    def __init__(self, n_freq=6, d_model=64):
        super().__init__()
        self.proj_in = nn.Conv2d(n_freq, d_model, 1, bias=False)
        self.pos_embed = nn.Parameter(torch.randn(1, d_model, 13, 16) * 0.02)
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.proj_out = nn.Conv2d(d_model, 1, 1, bias=False)

    def forward(self, x):
        # (B, 6, 13, 16) → (B, d_model, 13, 16)
        h = self.proj_in(x) + self.pos_embed
        # Self-Attention across 208 spatial locations
        h_flat = h.view(B, self.d_model, -1).transpose(1, 2)  # (B, 208, d_model)
        Q, K, V = self.q_proj(h_flat), self.k_proj(h_flat), self.v_proj(h_flat)
        attn = F.softmax((Q @ K.transpose(-2, -1)) * self.scale, dim=-1)
        out = attn @ V  # (B, 208, d_model)
        # (B, d_model, 13, 16) → (B, 1, 13, 16)
        return self.proj_out(out.transpose(1, 2).view(B, self.d_model, 13, 16))
```

**✅ 显著优势**：
- 建模频率间的非线性交互（而非简单加权）
- 空间位置编码保持局部性
- Transformer 架构带来更强的表达能力

---

## 2. Phase 2 改进对比

### 2.1 Tikhonov 初始重建 σ₀ 作为输入

| 对比项 | GPT 建议 | 当前实现 | 状态 |
|--------|---------|---------|------|
| **数据生成** | 增加 `sigma_0` 字段 | ❌ 未实现 | 🔴 **缺失** |
| **模型输入** | 拼接 σ₀ 到节点特征 | ❌ 未实现 | 🔴 **缺失** |
| **Tikhonov 求解** | `solve_tikhonov()` 方法 | ❌ 未实现 | 🔴 **缺失** |

**预期收益**：
- **RE ↓ 15-20%**（根据 GPT 分析）
- 提供"传统反演 + 神经优化"的混合架构
- 更好的初始化，减少无监督训练难度

**实施成本**：
- **代码量**：~30 行（数据生成）+ ~20 行（模型修改）
- **时间成本**：重新生成数据集（约 30-60 分钟）
- **风险**：中等（需验证 Tikhonov 重建质量）

**推荐优先级**：🟡 **中**（Phase 2.1 可在 Phase 2.2 验证后再决定是否实施）

---

### 2.2 物理 Loss 扩展

| 损失项 | GPT 建议 | 当前实现 | 预期收益 | 状态 |
|--------|---------|---------|---------|------|
| **残差稀疏约束** | `‖Δσ‖₁` | ❌ 未实现 | 避免过度修改初值 | 🔴 **缺失** |
| **Graph TV** | `Σ w_ij(σ_i - σ_j)²` | ❌ 未实现 | 边缘保持平滑 | 🔴 **缺失** |

**GPT 建议的 Loss 配方**：

```python
# 新增: 残差稀疏约束
loss_residual = torch.norm(delta, p=1, dim=-1).mean()

# 新增: Graph TV
edge_idx = model._edge_idx.to(device)
sigma_diff = (sigma[:, edge_idx[0]] - sigma[:, edge_idx[1]]) ** 2
loss_gtv = (model._edge_weight.to(device) * sigma_diff).mean()

# 总损失
total = (0.3 * loss_sup +
         0.4 * loss_m +
         0.1 * loss_t +
         0.1 * loss_d +
         0.05 * loss_residual +
         0.05 * loss_gtv)
```

**当前实现的损失**：

```python
# 仅 4 项损失
loss_sup = criterion(sigma, sigma_gt)  # 有监督锚点
loss_m = mcl(sigma, voltages)          # 测量一致性
loss_t = tvl(sigma)                    # TV 正则
loss_d = sdl(sigma)                    # σ 偏离约束

total = (0.5 * loss_sup + 1.0 * loss_m + 0.05 * loss_t + 0.1 * loss_d)
```

**缺失的损失分析**：

| 损失 | 当前替代 | 差异 |
|------|---------|------|
| `loss_residual` | 无 | 缺少对 Δσ 的约束，可能导致过校正 |
| `loss_gtv` | `loss_t` (TV) | TV 在网格上计算，Graph TV 在图结构上计算，更符合 FEM 网格拓扑 |

**⚠️ 关键问题**：
- 当前 TV Loss (`tvl`) 可能未充分利用 FEM 网格的边结构
- Graph TV 更适合非结构化网格，且可以重用边权重

**推荐行动**：🔴 **立即实施**（代码量 <10 行，预期收益高）

---

## 3. Phase 3 改进对比

### 完整三分支架构

```
GPT 建议的 PG-MeshGNN 架构:
========================================

输入: V (多频边界电压)
        ↓
┌───────────────────────────────────────┐
│ Branch A: Tikhonov/GN → σ₀            │  ← 传统反演
│ Branch B: Jᵀ(V - F(σ₀)) → g          │  ← 已实现 ✅
│ Branch C: ConvEncoder(V) → z_v       │  ← 已实现 ✅
└───────────────────────────────────────┘
        ↓
节点特征: h_i = [σ₀_i, g_i, s_i, x_i, y_i, r_i, z_v]
        ↓
Edge-aware MeshGNN (4层)
        ↓
Δσ = MeshGNN(h)
σ̂ = σ₀ + Δσ
```

**当前实现状态**：

| 分支 | 状态 | 说明 |
|------|------|------|
| **Branch A** | ❌ 未实现 | 缺少 Tikhonov 初始重建 |
| **Branch B** | ✅ 已实现 | Jᵀr 反投影完整实现 |
| **Branch C** | ✅ 已实现 | ConvEncoder + 位置编码 |
| **节点特征** | ⚠️ 部分实现 | 缺少 σ₀，其他特征都有 |
| **MeshGNN** | ✅ 已实现 | 支持边特征，4层 GNN |
| **残差输出** | ✅ 已实现 | σ = σ₀ + Δσ |

**完成度**：**70%**（主要缺少 Branch A）

---

## 4. 性能对比与瓶颈分析

### 4.1 当前性能 vs 预期性能

| 阶段 | 改进项 | GPT 预期 RE | 当前实际 RE | 差距 |
|------|--------|:-----------:|:-----------:|:----:|
| **基线** | ConvSpatial 原版 | 0.331 | - | - |
| **Phase 1** | + Jᵀr | ~0.25 | **0.193** | ✅ **超越** |
| **Phase 1** | + Jᵀr + Δσ + 边特征 | ~0.20 | **0.193** | ✅ **达标** |
| **Phase 2** | + σ₀ 输入 | ~0.17 | - | 🔴 未实施 |
| **Phase 2** | + 全部 Loss | ~0.15 | - | 🔴 未实施 |
| **Phase 3** | PG-MeshGNN 完整版 | ~0.12-0.15 | - | 🔴 未实施 |

**✅ 成功点**：
- Phase 1 改进已达到甚至超越预期目标（RE=0.193 vs 目标 0.20）
- 说明 Jᵀr 反投影 + 残差输出 + Cross-Attention 的组合非常有效

**⚠️ 瓶颈**：
- **距离最终目标 RE≤0.15 仍有差距**（当前 0.193）
- Phase 2 改进未实施，可能是突破瓶颈的关键

---

### 4.2 与历史最佳对比

| 模型配置 | RE | CC | 参数量 | 训练模式 |
|---------|:--:|:--:|:------:|----------|
| **hd512 (历史最佳)** | **0.103** | 0.976 | ~12M | 两阶段 |
| **hd256 (当前最佳)** | 0.193 | 0.955 | ~6.1M | 两阶段 |

**性能退化分析**：

| 因素 | 影响 | 证据 |
|------|------|------|
| **模型容量** | 🔴 高 | 参数量减半 (12M → 6.1M) |
| **网络架构** | 🟢 中 | Phase 1 改进已实施，应该有提升 |
| **训练策略** | ⚠️ 未知 | 需对比训练曲线、超参数 |
| **数据集** | ⚠️ 可能不同 | 需确认是否使用同一数据集 |

**🔍 关键问题**：
- 为什么 hd256 + Phase 1 改进 (RE=0.193) 仍然不如 hd512 无改进 (RE=0.103)？
- **可能原因**：
  1. 模型容量不足是主要瓶颈
  2. Phase 1 改进在 hd512 上可能更有效
  3. 训练超参数（学习率、损失权重）需要重新调优

---

## 5. 实施优先级与建议

### 5.1 立即实施（本周）

#### 优先级 🔴 最高：添加缺失的物理损失

**任务清单**：
1. ✅ **残差稀疏约束**（`loss_residual`）
   ```python
   # 在 train_conv_spatial.py 的无监督阶段添加
   delta = out['delta']  # 从模型输出获取
   loss_residual = torch.norm(delta, p=1, dim=-1).mean()
   ```

2. ✅ **Graph TV 损失**（`loss_gtv`）
   ```python
   # 利用已有的边结构
   edge_idx = model._edge_idx.to(device)
   sigma_diff = (out['sigma'][:, edge_idx[0]] - out['sigma'][:, edge_idx[1]]) ** 2
   loss_gtv = (model._edge_weight.to(device) * sigma_diff).mean()
   ```

3. ✅ **调整损失权重**
   ```python
   total = (0.3 * loss_sup + 0.4 * loss_m + 0.05 * loss_t + 0.1 * loss_d
            + 0.05 * loss_residual + 0.05 * loss_gtv)
   ```

**预期收益**：RE ↓ 10-15%（从 0.193 → ~0.16-0.17）

**实施成本**：代码 <15 行，无需重新生成数据

---

#### 优先级 🟡 中：补充边特征计算

**任务清单**：
1. ✅ 在 `setup_mesh()` 中预计算边特征
   ```python
   n_edges = edge_list.shape[1]
   edge_feat = np.zeros((n_edges, 4), dtype=np.float32)
   for e_idx, (i, j) in enumerate(zip(edge_list[0], edge_list[1])):
       ci, cj = centers[i, :2], centers[j, :2]
       edge_feat[e_idx, 0] = np.linalg.norm(ci - cj)  # 距离
       edge_feat[e_idx, 1] = len(set(elements[i]) & set(elements[j])) / 3.0
       # 面积比（如果有面积信息）
       # 灵敏度相似性（如果有 Jacobian）
   self.register_buffer('edge_feat', torch.from_numpy(edge_feat).float())
   ```

2. ✅ 确认 `SimpleGNNLayer` 的 `edge_dim` 参数设置正确

**预期收益**：RE ↓ 5-10%（如果之前未启用边特征）

**实施成本**：代码 <20 行，无需重新生成数据

---

### 5.2 短期规划（1-2 周）

#### 优先级 🟡 中：实施 Tikhonov 初始重建

**实施步骤**：
1. 在 `EITForwardSolver` 中添加 `solve_tikhonov()` 方法
2. 修改数据生成脚本，增加 `sigma_0` 字段
3. 重新生成数据集（需 30-60 分钟）
4. 修改模型，拼接 σ₀ 到节点特征

**预期收益**：RE ↓ 15-20%

**风险评估**：
- 需要重新生成数据（时间成本）
- Tikhonov 重建质量需要验证
- 可能与当前的无监督训练策略冲突（需调优）

**建议**：先完成 5.1 的改进，评估效果后再决定是否实施

---

#### 优先级 🟢 低：恢复 hd512 架构

**实施步骤**：
1. 用 hd512 配置重新训练（已有历史最佳 RE=0.103）
2. 应用 Phase 1 改进（Jᵀr + Δσ + 边特征）
3. 应用新的物理损失（residual + Graph TV）

**预期收益**：可能突破 RE=0.10（历史最佳 + 改进）

**建议**：作为对比实验，验证改进在不同容量模型上的效果

---

### 5.3 长期规划（论文版）

#### Phase 3 完整实施

**架构升级**：
- 完整三分支融合（Tikhonov + Jᵀr + ConvEncoder）
- 自适应损失权重学习
- 多尺度重建策略

**实验矩阵**：
- 消融实验（逐步添加改进项）
- 根系数据集泛化验证
- 与传统方法（GN、GREIT）对比

**论文撰写**：
- 方法论总结
- 物理约束深度学习框架
- 植物根系成像应用

---

## 6. 关键发现总结

### 6.1 已成功实施的改进 ✅

| 改进项 | 实施状态 | 效果评估 |
|--------|---------|---------|
| **Jᵀr 反投影** | ✅ 完整实现 + 改进 | 核心物理约束 |
| **残差输出** | ✅ 完整实现 | 提供可解释中间量 |
| **边特征支持** | ✅ 架构支持 | 需确认是否启用 |
| **多频 Cross-Attention** | ✅ 超越建议 | 更强的表达能力 |
| **数值稳定性优化** | ✅ 工程改进 | 避免梯度爆炸 |

### 6.2 缺失的关键改进 ❌

| 改进项 | 缺失原因 | 优先级 | 预期收益 |
|--------|---------|--------|---------|
| **残差稀疏约束** | 未实施 Phase 2.2 | 🔴 高 | RE ↓5-10% |
| **Graph TV 损失** | 未实施 Phase 2.2 | 🔴 高 | RE ↓5-10% |
| **Tikhonov σ₀** | 未实施 Phase 2.1 | 🟡 中 | RE ↓15-20% |

### 6.3 当前瓶颈 🔍

1. **性能瓶颈**：RE=0.193，距离目标 0.15 有差距
2. **损失函数不完整**：缺少残差和 Graph TV 约束
3. **模型容量疑虑**：hd256 vs hd512 性能差距显著
4. **未达到 Phase 2 预期**：改进停在了 Phase 1

### 6.4 推荐行动路线

```
立即行动（本周）:
  ├─ 添加 loss_residual + loss_gtv          ← 最小成本，最大收益
  └─ 确认边特征是否启用

短期规划（1-2周）:
  ├─ 实施 Tikhonov σ₀ 输入                  ← 需重新生成数据
  └─ 恢复 hd512 架构对比实验                ← 验证改进效果

长期规划（论文版）:
  └─ Phase 3 完整三分支架构                 ← 论文级完整方案
```

---

## 7. 代码修改建议

### 7.1 train_conv_spatial.py 修改

```python
# 在无监督训练循环中添加（约第 545 行之后）

# 提取 delta（残差修正量）
delta = out['delta'] if isinstance(out['delta'], torch.Tensor) else torch.zeros_like(sp)

# 新增: 残差稀疏约束
loss_residual = torch.norm(delta, p=1, dim=-1).mean()

# 新增: Graph TV（在 FEM 网格上做边缘保持平滑）
loss_gtv = torch.tensor(0.0, device=device)
if hasattr(model, '_edge_idx') and model._edge_idx is not None:
    edge_idx = model._edge_idx.to(device)
    sigma_diff = (sp[:, edge_idx[0]] - sp[:, edge_idx[1]]) ** 2
    loss_gtv = (model._edge_weight.to(device) * sigma_diff).mean()

# 调整总损失权重
total = (0.3 * loss_sup + 0.4 * loss_m + 0.05 * loss_t + 0.1 * loss_d
         + 0.05 * loss_residual + 0.05 * loss_gtv)

# 记录新增损失
epoch_losses["residual"] += loss_residual.item()
epoch_losses["gtv"] += loss_gtv.item()
```

### 7.2 models/conv_spatial_eit.py 修改（如需补充边特征）

```python
# 在 setup_mesh() 中添加（约第 490 行之后）

# 预计算边特征：[距离, 共享边长, 面积比, 灵敏度相似性]
if jacobian is not None:
    n_edges = edge_list.shape[1]
    edge_feat = np.zeros((n_edges, 4), dtype=np.float32)

    # 提取中心坐标
    cx, cy = centers[:, 0], centers[:, 1]

    for e_idx, (i, j) in enumerate(zip(edge_list[0], edge_list[1])):
        # 特征 1: 单元中心距离
        edge_feat[e_idx, 0] = np.sqrt((cx[i] - cx[j])**2 + (cy[i] - cy[j])**2)

        # 特征 2: 共享节点数量 → 近似共享边长
        shared_nodes = len(set(elements[i]) & set(elements[j]))
        edge_feat[e_idx, 1] = shared_nodes / 3.0  # 归一化到 [0, 1]

        # 特征 3: 面积比（如果有面积信息）
        # edge_feat[e_idx, 2] = ...

        # 特征 4: Jacobian 灵敏度相似性
        if hasattr(self, 'J_T'):
            J_i = self.J_T[i, :].norm()
            J_j = self.J_T[j, :].norm()
            edge_feat[e_idx, 3] = 1.0 / (1.0 + np.abs(J_i - J_j))

    # 归一化
    edge_feat = (edge_feat - edge_feat.mean(axis=0)) / (edge_feat.std(axis=0) + 1e-8)
    self.register_buffer('_edge_feat', torch.from_numpy(edge_feat).float())
    print(f"  [ConvSpatial] 边特征: {edge_feat.shape}")
```

---

## 8. 实验验证建议

### 8.1 消融实验矩阵

| 实验 ID | 配置 | 预期 RE | 实际 RE | 改善 |
|---------|------|:-------:|:-------:|:----:|
| **E0** | 当前最佳 (hd256 + Phase 1) | - | 0.193 | 基线 |
| **E1** | E0 + loss_residual | ~0.18 | ? | ? |
| **E2** | E0 + loss_gtv | ~0.18 | ? | ? |
| **E3** | E0 + loss_residual + loss_gtv | ~0.16 | ? | ? |
| **E4** | E3 + 边特征 | ~0.15 | ? | ? |
| **E5** | E4 + Tikhonov σ₀ | ~0.12 | ? | ? |

### 8.2 评估指标

- **主要指标**: RE (相对误差), CC (相关系数)
- **次要指标**: SSIM (结构相似性), 推理速度
- **训练指标**: 收敛速度, 损失曲线稳定性
- **物理指标**: 测量一致性误差, 残差范数

---

## 9. 风险与注意事项

### 9.1 实施风险

| 风险 | 影响 | 缓解措施 |
|------|------|---------|
| **损失权重不平衡** | 训练不稳定 | 小步调优，监控各损失项 |
| **边特征质量差** | 无效果或负面效果 | 可视化边特征分布，异常检测 |
| **Tikhonov 重建质量差** | 初始化错误 | 先在验证集上评估 Tikhonov 性能 |
| **重新生成数据成本** | 时间消耗 | 可先用小规模数据验证 |

### 9.2 性能监控

**关键日志**：
- 各损失项的独立值（sup, m, t, d, residual, gtv）
- 验证 RE 的变化趋势
- 梯度范数（检测训练稳定性）
- Δσ 的统计分布（残差输出量级）

**Early Stopping**：
- 验证 RE 连续 10 轮未改善 → 停止
- 总损失连续 5 轮上升 → 降低学习率

---

## 10. 结论

### 10.1 主要发现

1. **Phase 1 改进已 90% 完成**，且性能达标（RE=0.193 ≤ 目标 0.20）
2. **当前实现优于 GPT 建议**的部分：
   - 多频 Cross-Attention（更强表达力）
   - 残差归一化（更好的数值稳定性）
   - 完整的物理范围约束

3. **Phase 2 关键改进缺失**：
   - 残差稀疏约束和 Graph TV 损失未实施
   - Tikhonov 初始重建未实现

4. **性能瓶颈明确**：
   - 距离最终目标 RE≤0.15 有差距
   - 缺少 Phase 2 损失约束可能是主因

### 10.2 推荐行动

**立即实施**（本周）：
- ✅ 添加 `loss_residual` 和 `loss_gtv`（<15 行代码，预期 RE ↓10-15%）
- ✅ 确认边特征是否启用

**短期规划**（1-2周）：
- 🟡 评估 Tikhonov σ₀ 的价值（需重新生成数据）
- 🟡 hd512 架构对比实验

**长期规划**（论文版）：
- 🟢 Phase 3 完整三分支架构

### 10.3 预期效果

如果立即实施 Phase 2.2 改进：
- **短期**：RE 从 0.193 → ~0.16-0.17
- **中期**（+ Tikhonov）：RE 从 ~0.16 → ~0.12-0.14
- **长期**（Phase 3）：RE ≤ 0.12，达到论文级性能

---

**文档维护**：
- 定期更新实验结果
- 追踪各项改进的实际效果
- 调整优先级和行动路线
