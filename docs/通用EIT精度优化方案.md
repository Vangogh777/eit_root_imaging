# Conv-Spatial EIT 通用精度优化方案

> **目标**：在现有 `ConvSpatialEIT`（Conv2D 编码 → Grid Sampling → GNN 精修）基础上，
> 系统性修复影响精度的瓶颈，打造**高精度通用 EIT 算法**（不限根成像场景）。
>
> **范围**：仅聚焦通用 EIT 重建精度，暂不考虑植物根方向的多频 Cole-Cole 物理扩展。
>
> **关联文件**：
> - `train_conv_spatial.py` — 训练入口
> - `models/conv_spatial_eit.py` — 网络定义
> - `training/loss.py` — 损失函数
> - `data/eit_forward.py` — FEM 正问题求解
> - `data/datasets/eit_dataset.py` — 数据加载
> - `data/generate_circle_dataset.py` — 数据生成（域随机化）

> **修订记录**：v1.1 — 采纳评审意见后的调整：
> - 🔴 新增 P0：MCL 走 Jacobian 模式导致梯度偏差（实际 bug，必须修）
> - 🟠 Conv2D 错配**简化**为只做输入归一化（Conv 提模式、位置靠 GridSampling+GNN+位置编码吸收）
> - 🟠 GridSampler 方案改为**保持 13×16 原生分辨率不插值**（避免上采样信息损失）
> - 🟠 **域随机化从 P3 升到 P1**（sim-to-real gap 才是通用化的真正天花板）
> - 🟡 Jacobian 先验注入**降为可选**（Jacobian 自身精度存疑时收益打折）
> - 📅 实施顺序重排，按"修 bug → 修架构 → 修泛化"递进

---

## 目录

- [一、问题总览（优先级矩阵）](#一问题总览优先级矩阵)
- [二、P0 致命问题（必改）](#二p0-致命问题必改)
- [三、P1 高收益问题](#三p1-高收益问题)
- [四、P2 训练流程优化](#四p2-训练流程优化)
- [五、P3 架构演进方向（可选）](#五p3-架构演进方向可选)
- [六、实施路线图](#六实施路线图)

---

## 一、问题总览（优先级矩阵）

| 编号 | 优先级 | 问题 | 所在文件 | 预期收益 |
|:---:|:---:|---|---|---|
| 0 | 🔴 P0 | **MCL 走 Jacobian 模式 → 梯度偏差**（实际 bug） | `train_conv_spatial.py:295` | 修复错误梯度 |
| 1 | 🔴 P0 | 多频率电压完全相同（数据退化） | `data/eit_forward.py:149` | 解锁真实信息量 |
| 2 | 🔴 P0 | GNN 缺位置编码，中心/边缘无法区分 | `models/conv_spatial_eit.py:147,291` | 中心重建清晰 |
| 3 | 🟠 P1 | 输入电压未归一化（Conv 错配的最小修复） | `models/conv_spatial_eit.py:291` | 训练更稳 |
| 4 | 🟠 P1 | GridSampler 分辨率太粗(8×8) | `models/conv_spatial_eit.py:87` | 空间细节保留 |
| 5 | 🟠 P1 | TV 用 KNN 邻接而非真实网格边 | `training/loss.py:171` | TV 物理正确 |
| 6 | 🟠 P1 | SmoothnessLoss 用数组索引差（无意义） | `training/loss.py:297` | 去除错误正则 |
| 7 | 🟠 P1 | SigmaDeviationLoss 与 sigmoid 输出冲突 | `train_conv_spatial.py:324` | 避免对抗监督 |
| 8 | 🟠 P1 | **域随机化缺失（sim-to-real gap）** | `data/generate_circle_dataset.py` | 泛化能力提升 |
| 9 | 🟡 P2 | BatchSampler 迭代逻辑有缺陷 | `train_conv_spatial.py:132` | 训练稳定性 |
| 10 | 🟡 P2 | LR scheduler 的 T_max 与实际步数不符 | `train_conv_spatial.py:180` | LR 曲线正确 |
| 11 | 🟡 P2 | 验证指标只有 RE，缺 SSIM/IoU | `train_conv_spatial.py:252` | 评估更全面 |
| 12 | 🟡 P2 | 缺少 EMA 模型权重 | — | +1~3% RE |
| 13 | 🟢 P3 | 缺少 TTA（旋转对称增强） | — | 降噪声敏感度 |
| 14 | 🟢 P3 | Jacobian 先验注入 | — | 收益取决于 Jacobian 精度 |

---

## 二、P0 致命问题（必改）

### 问题 0：MCL 走 Jacobian 模式 → 无监督梯度偏差（实际 bug）⚠️

**位置**：`train_conv_spatial.py:295-300`

**现状**：
```python
mcl = MeasurementConsistencyLoss(
    mode='jacobian' if jacobian is not None else 'full_fem',
    jacobian=jacobian,
    sigma_ref_value=0.01,
    forward_solver=lambda s: solver.solve_multi_frequency(s),
)
```

只要 `data/generated/jacobian.npy` 存在（实际存在），就会走**纯 Jacobian 模式**。

**影响（物理问题）**：

Jacobian `J = dV/dσ` 是在 `σ_ref = 0.01`（土壤背景）处**线性化**的灵敏度矩阵。
但本场景内含物 σ=0.05，对比度 5×，**线性近似在 σ>0.03 时误差达 10-30%**：
```
真实关系:  V = F(σ)              （非线性）
线性近似:  V ≈ V_ref + J·(σ-σ_ref)  （在 σ_ref 附近才准）
```

走纯 `jacobian` 模式时，无监督阶段的梯度回传用的是**错误梯度**——
模型在努力最小化 `‖J·(σ_pred - 0.01) - V_measured‖²`，但这个目标本身就偏离真实物理。
**这是无监督精度的硬天花板。**

**修复方案**：强制走 `full_fem` 模式（`loss.py:94-126` 已实现：FEM 正解算真实 V，Jacobian 仅作可微分辅助梯度）。

修改 `train_conv_spatial.py`：
```python
mcl = MeasurementConsistencyLoss(
    mode='full_fem',               # ★ 强制 full_fem，不要根据 jacobian 自动选
    jacobian=jacobian,             # 保留，作为辅助梯度（loss 内部 loss_aux）
    sigma_ref_value=0.01,
    forward_solver=lambda s: solver.solve_multi_frequency(s),
    fem_interval=5,                # 每 5 步跑一次完整 FEM（默认值，平衡速度精度）
)
```

**注意**：`full_fem` 模式比 `jacobian` 慢（每 `fem_interval` 步要跑 B 次 FEM 正解）。
如果训练速度不可接受，可调大 `fem_interval`（如 10）。

> **残余问题**：`full_fem` 模式的可微分梯度**仍通过 Jacobian 回传**（`loss.py:117-122`），
> 只是把真实 V 作为监督目标。要彻底消除梯度偏差，需要**可微 FEM** 或**迭代重线性化**，
> 属于 P3 架构演进范畴（见第五节方向 C）。

---

### 问题 1：多频率电压完全相同 ⚠️ 最严重

**位置**：`data/eit_forward.py:149-162`

**现状**：
```python
def solve_multi_frequency(self, sigma):
    v = self.solve(sigma)
    return np.tile(v, (len(self.frequencies), 1))  # ← 6个频率复制成6份相同电压
```

**影响**：数据集里 6 个频率的电压**完全相同**。但 Conv encoder 用了 6 通道去处理它 →
6 个通道学到同一信号，**模型容量严重浪费**，多频融合的优势完全消失。实际独立信息只有 208 个测量值，但模型以为有 6×208。

**修复方案**（暂不做 Cole-Cole，采用最小改动）：

**方案 A — 退化为单频（推荐，最诚实）**：

把 6 频率降到 1 频率，Conv encoder 改为 1 通道输入，模型更精简。修改 `models/conv_spatial_eit.py`：
```python
# ConvSpatialEIT.__init__
self.encoder = ConvEncoder(in_channels=1, base_ch=48)   # 6 → 1

# forward
def forward(self, voltages):
    if voltages.dim() == 3:           # (B, 6, 208)
        voltages = voltages[:, :1, :] # 取第一个（或平均）
    x = voltages.view(B, 1, 13, 16)   # (B, 1, 13, 16)
```

**方案 B — 多频做数据增强**（保留接口）：
保留 6 通道输入，但在数据生成时给每个频率注入**不同随机噪声**，让网络学习对噪声鲁棒的多频一致性：
```python
# generate_circle_dataset.py: _generate_one 内
V = _solver.solve_multi_frequency(sigma)
# 给每个频率独立加噪（而不是整个 V 一次性加噪）
for fi in range(V.shape[0]):
    V[fi] = _solver.add_noise(V[fi:fi+1], rng.uniform(-40, -20))[0]
```

> **建议**：先采用方案 A 让模型诚实，后续真做多频物理时再恢复 6 通道。

---

### 问题 2：GNN 缺位置编码，无法区分中心/边缘单元

**位置**：
- `models/conv_spatial_eit.py:118-144`（GridSampler）
- `models/conv_spatial_eit.py:147-191`（SimpleGNNLayer）
- `models/conv_spatial_eit.py:291-333`（forward）

**现状**：
1. GridSampler 从 8×8=64 像素采样到 ~11466 个单元 → **大量单元采到完全相同的特征**
2. GNN 节点初始特征只有 feature，**没有坐标/位置信息**
3. EIT 对中心区域天然不敏感（灵敏度低），但模型没有任何机制告诉它"这是边缘单元 / 这是中心单元"

**修复方案**：在 `setup_mesh` 预计算位置编码，在 `forward` 拼接到节点特征。

修改 `models/conv_spatial_eit.py`：
```python
class ConvSpatialEIT(nn.Module):
    def setup_mesh(self, centers, elements):
        # ... 原有代码（构建 edge_idx, edge_weight）...

        # ★ 新增：节点位置编码
        # 1. 归一化坐标 (n_elems, 2)
        c = centers[:, :2].astype(np.float32)
        r_max = np.abs(c).max() + 1e-8
        pos = c / r_max                                       # (N, 2) ∈ [-1, 1]
        # 2. 半径编码（区分中心/边缘，EIT 灵敏度差异巨大）
        radius = np.linalg.norm(c, axis=1, keepdims=True) / r_max  # (N, 1)
        # 3. Fourier 位置编码（提升表达能力）
        def fourier(x, n_freq=8, scale=2.0):
            freqs = scale ** torch.arange(n_freq).float() * np.pi  # (n_freq,)
            args = torch.from_numpy(x).float()[:, :, None] * freqs  # (N, 2, n_freq)
            return torch.cat([torch.sin(args), torch.cos(args)], -1).reshape(len(x), -1)
        pe = torch.cat([fourier(pos), torch.from_numpy(radius).float()], dim=-1)
        self.register_buffer('pos_encoding', pe)  # (N, pos_dim)

        # ★ GNN 输入维度 = Conv 通道 + 位置编码维度
        self.gnn_in_dim = 128 + pe.shape[1]
        # 重建 gnn_blocks（或者改用动态拼接 Linear）
        ...

    def forward(self, voltages):
        ...
        node_feat = self.sampler(feat)                # (B, N, 128)
        # ★ 拼接位置编码
        pe = self.pos_encoding.unsqueeze(0).expand(B, -1, -1)  # (B, N, pos_dim)
        node_feat = torch.cat([node_feat, pe], dim=-1)          # (B, N, 128+pos_dim)
        # 后续 GNN 用 gnn_in_dim 作为第一层输入维度
        ...
```

**注意**：第一层 GNN 的 `in_dim` 需要从 128 改为 `128 + pos_dim`，更新 `gnn_blocks` 构造逻辑。

---

## 三、P1 高收益问题

### 问题 3：GridSampler 分辨率太粗（8×8）+ 不必要的插值

**位置**：`models/conv_spatial_eit.py:75-88`

**现状**：Conv encoder 输出经 stage2 的 stride=2 下采样到 7×8，再 `F.interpolate` 到 `(8, 8) = 64` 像素，
**平均 ~180 个单元共享一个像素的特征** → 空间细节严重丢失。

**修复方案**：去掉下采样 + 去掉插值，**保持 Conv 输入的原生 13×16 分辨率**直接做 Grid Sampling。

> ⚠️ **评审修订**：原方案是插值到 16×16，但**上采样本身就是信息损失**——
> 13×16→16×16 是凭空造像素，纯增噪。不如直接用 encoder 原生分辨率 13×16 = 208 像素采样，
> 比插值更自然、也更忠实于输入信息量。

```python
class ConvEncoder(nn.Module):
    def __init__(self, in_channels=1, base_ch=48):   # 配合问题1：单频 in=1
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, base_ch, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(base_ch), nn.ReLU(inplace=True),
        )
        self.stage1 = nn.Sequential(
            nn.Conv2d(base_ch, base_ch*2, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(base_ch*2), nn.ReLU(inplace=True),
            ResBlock(base_ch*2), ResBlock(base_ch*2),
        )
        # ★ stage2 删除 stride=2，保持分辨率
        self.stage2 = nn.Sequential(
            nn.Conv2d(base_ch*2, base_ch*4, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(base_ch*4), nn.ReLU(inplace=True),
            ResBlock(base_ch*4),
        )
        self.out_proj = nn.Sequential(
            nn.Conv2d(base_ch*4, 128, 1, bias=False),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.stem(x)     # (B, 48, 13, 16)
        x = self.stage1(x)   # (B, 96, 13, 16)
        x = self.stage2(x)   # (B, 192, 13, 16) -- 不下采样
        x = self.out_proj(x) # (B, 128, 13, 16)
        # ★ 删除 F.interpolate，直接返回原生分辨率
        return x             # (B, 128, 13, 16) -- 208 像素
```

**配套**：GridSampler 的 `setup_grid` 不需改（它用单元中心坐标去 `grid_sample` 特征图，
特征图从 (8,8) 变成 (13,16) 自动适配）。GridSampler 归一化时用 (13,16) 的边界即可。

---

### 问题 5：TV 用 KNN 邻接而非真实网格边

**位置**：`training/loss.py:171-196`（`TVRegularizationLoss._compute_edge_weights`）

**现状**：用 `scipy.cKDTree` 找 K=8 近邻作为 TV 边。
**问题**：KNN 邻居可能跨越物理边界（环状域里选到对岸的单元），TV 梯度方向错误。

**修复方案**：复用模型 `setup_mesh` 已经建好的**真实 FEM 邻接**（基于共享节点）。

修改 `training/loss.py`：
```python
class TVRegularizationLoss(nn.Module):
    def __init__(self, mesh_elements: torch.Tensor, mesh_nodes: torch.Tensor,
                 element_centers: Optional[torch.Tensor] = None):
        """
        ★ 改用真实 FEM 邻接（基于共享节点）
        不再依赖 KNN
        """
        super().__init__()
        self.edge_idx = self._build_fem_adjacency(mesh_elements)  # (2, n_edges)

    @staticmethod
    def _build_fem_adjacency(elements):
        """从三角单元构建共享边邻接"""
        from collections import defaultdict
        node_to_elems = defaultdict(list)
        for i, tri in in enumerate(elements.tolist()):
            for nd in tri:
                node_to_elems[nd].append(i)
        edges = set()
        for elems in node_to_elems.values():
            for a in range(len(elems)):
                for b in range(a+1, len(elems)):
                    i, j = elems[a], elems[b]
                    edges.add((min(i, j), max(i, j)))
        return torch.tensor(list(edges), dtype=torch.long).T  # (2, n_edges)

    def forward(self, sigma):
        # ... 不变，但 edge_idx 现在是物理邻接 ...
        sigma_diff = sigma[:, self.edge_idx[0]] - sigma[:, self.edge_idx[1]]
        return torch.abs(sigma_diff).mean()
```

**同步修改** `train_conv_spatial.py:301-305`，构造时传入 `mesh_elements`。

---

### 问题 6：SmoothnessLoss 用数组索引差（无意义）

**位置**：`training/loss.py:297-306`（`SmoothnessLoss.forward`）

**现状**：
```python
diff = sigma[:, 1:] - sigma[:, :-1]   # 相邻"数组索引"的差
```

**问题**：FEM mesh 单元的**数组索引顺序与空间位置完全无关**（mesh 生成时编号是任意的）。
这个 loss 在惩罚**编号相邻单元的电导率差**，**完全没有物理意义**，相当于随机噪声正则。

**修复方案**：

**方案 A（推荐）：直接删除**。TV 正则（修正后）已经覆盖了空间平滑约束，SmoothnessLoss 是冗余且错误的。

**方案 B：改用 Laplacian 正则**（如果想保留一个独立于 TV 的高阶平滑约束）：
```python
class LaplacianSmoothnessLoss(nn.Module):
    """基于真实邻接的二阶平滑（Laplacian）"""
    def __init__(self, edge_idx: torch.Tensor):
        self.edge_idx = edge_idx  # (2, n_edges)，复用 TV 的 FEM 邻接
    def forward(self, sigma):
        # 一阶差分的方差（鼓励均匀平滑，不只是稀疏边缘）
        diff = sigma[:, self.edge_idx[0]] - sigma[:, self.edge_idx[1]]
        return diff.pow(2).mean()  # L2 版本，与 TV(L1) 互补
```

**同步修改** `train_conv_spatial.py:307,324` 删除或替换 `sml`。

---

### 问题 7：SigmaDeviationLoss 与 sigmoid 输出冲突

**位置**：
- `models/conv_spatial_eit.py:326`：`sigma = sigmoid(raw) * (0.1-0.005) + 0.005` → 输出范围 `[0.005, 0.1]`
- `train_conv_spatial.py:306,324`：`SigmaDeviationLoss(sigma_ref_value=0.01)` + 权重 0.1

**现状**：内含物 GT 是 0.05，但 `SigmaDeviationLoss` 在拉它回到 0.01 → **对抗监督信号**。
该 loss 在无监督阶段才需要（约束 Jacobian 线性化点附近），权重 0.1 偏高。

**修复方案**：
1. **有监督阶段关闭它**（它只对无监督有意义）
2. **无监督阶段大幅降权**（0.1 → 0.01），或改成**只对低电导率区域**惩罚偏离：
```python
# 修改 SigmaDeviationLoss.forward：只对背景区域（sigma_pred 较小）约束
class SigmaDeviationLoss(nn.Module):
    def forward(self, sigma_pred):
        sigma_ref = self.sigma_ref_value
        # 软权重：sigma 越接近背景，惩罚越大；高电导率区域（内含物）豁免
        bg_weight = torch.exp(-((sigma_pred - sigma_ref) / 0.02).clamp(0, 4))
        diff = (sigma_pred - sigma_ref) * bg_weight
        return diff.pow(2).mean()
```

---

### 问题 8：输入电压未归一化（Conv 错配的最小修复）

> ⚠️ **评审修订**：原版"P0-2 Conv2D 物理错配"提出 Conv→Attention/Transformer 重构，
> **已被降级并简化**。原因：Conv encoder 提的是**模式特征**（哪些激励-测量组合相关），
> 真正的空间位置由 GridSampler 坐标 + GNN 拓扑 + 位置编码（P0-2）共同决定。
> 一旦位置编码落地，Conv 错配的影响被吸收，架构重构性价比低。
> **本项只做输入归一化即可。**

**位置**：`models/conv_spatial_eit.py:291`（`ConvSpatialEIT.forward`）

**现状**：输入电压直接 reshape 进 Conv，**未做归一化**。
不同样本的电压幅度差异巨大（取决于内含物大小/位置），导致 BN 层统计不稳定、学习困难。

**修复方案**：在 reshape 前按样本归一化：

```python
def forward(self, voltages):
    B = voltages.shape[0]
    if voltages.dim() == 3:
        x = voltages[:, :1, :].view(B, 1, 13, 16)   # 配合 P0-1 单频
    else:
        x = voltages
    # ★ 按样本归一化（去除整体幅度差异）
    amax = x.flatten(1).abs().max(dim=1)[0].view(B, 1, 1, 1) + 1e-8
    x = x / amax
    feat = self.encoder(x)
    ...
```

> 这是最小改动版的"P0-2 修复"。可选增强：除了 max 归一化，还可以同时 concat 一个
> **全局幅度特征**（`amax` 本身）作为额外通道，让模型保留幅度信息。

---

### 问题 9：域随机化缺失（sim-to-real gap）⚠️ 通用化的真正天花板

> 🆕 **评审升级**：原列在 P3，现升到 P1。
> 对"通用 EIT"目标，**sim-to-real gap 是比架构更重要的瓶颈**。

**位置**：`data/generate_circle_dataset.py`（数据生成）+ `data/eit_forward.py`（仿真器）

**现状**：
1. train/val/test 三集都来自**同一个 FEM 仿真器、同一套参数**（`generate_circle_dataset.py:149-151`，三集只差种子）
2. 仿真参数完全确定：电极位置固定、接触阻抗 0、系统增益 1、噪声范围固定
3. 模型学到的是**仿真器的偏置**而非**物理规律** → 仿真集 RE=0.05，真实数据可能 RE=0.5

**修复方案**：在数据生成时引入**域随机化**（domain randomization），让模型见过各种"非理想"情况：

修改 `data/generate_circle_dataset.py` 的 `_generate_one`：
```python
def _generate_one(seed):
    rng = np.random.RandomState(seed)
    cx, cy, r = _sample_position(rng)

    # ── 1. 电导率随机化（不只 0.01/0.05 两个值）──
    sigma_bg  = rng.uniform(0.005, 0.02)      # 土壤背景在范围内浮动
    sigma_inc = rng.uniform(0.03, 0.08)       # 内含物电导率浮动
    # 对比度也随机：1.5× ~ 8×

    sigma = np.full(n_elems, sigma_bg, dtype=np.float32)
    sigma[inside] = sigma_inc

    # ── 2. 求解 + 多源随机噪声 ──
    V = _solver.solve_multi_frequency(sigma)
    # 噪声范围扩大：从 (-40,-20) 扩到 (-50,-10)，覆盖真实仪器各种 SNR
    noise_db = rng.uniform(-50, -10)
    V_noisy = _solver.add_noise(V, noise_db)

    # ── 3. 系统增益随机扰动（每通道独立 ±5%）──
    gain = 1.0 + 0.05 * rng.randn(*V_noisy.shape)
    V_noisy = V_noisy * gain

    # ── 4. 接触阻抗漂移（模拟电极接触不良）──
    if rng.random() < 0.3:   # 30% 概率触发
        sigma_drift = _solver.simulate_contact_impedance_drift(
            sigma, drift_scale=rng.uniform(0.01, 0.05))
        V_noisy = _solver.solve_multi_frequency(sigma_drift)
        V_noisy = _solver.add_noise(V_noisy, noise_db)

    # ── 5. 基线偏移（模拟仪器零点漂移）──
    baseline = rng.uniform(-1e-4, 1e-4)
    V_noisy = V_noisy + baseline

    return {...}
```

> **可选进一步增强**（投入产出比稍低，P3 再考虑）：
> - 电极位置抖动（±2mm，需修改 mesh 重建，成本较高）
> - 通道缺失/坏通道模拟（随机把某些测量置 0）
> - 不同激励模式的混合训练

**配套**：重新生成数据集后，模型在仿真集 RE 可能**略升**（因为变难了），
但在真实数据上的泛化能力会**大幅改善**——这才是"通用 EIT"的关键。

---

## 四、P2 训练流程优化

### 问题 10：BatchSampler 迭代逻辑有缺陷

**位置**：`train_conv_spatial.py:132-153`

**现状**：用模运算 `% len(edge_idx)` 拼接 batch，导致：
- batch 大小不稳定（line 148 的截断会让某些 batch 少样本）
- epoch 内样本覆盖不均（边缘/中心被访问次数不平衡）
- `__len__` 与实际迭代数不一致（影响 scheduler / tqdm 进度条）

**修复方案**：改用更标准的平衡采样器：
```python
class BalancedBatchSampler(torch.utils.data.Sampler):
    def __init__(self, labels, batch_size, edge_ratio=0.5, seed=0):
        self.labels = np.asarray(labels)
        self.batch_size = batch_size
        self.n_edge_per = max(1, int(batch_size * edge_ratio))
        self.n_center_per = batch_size - self.n_edge_per
        self.edge_idx = np.where(labels == 1)[0]
        self.center_idx = np.where(labels == 0)[0]
        self.rng = np.random.RandomState(seed)

    def __iter__(self):
        # 用重复采样保证两个池都被充分访问
        n_batches = max(len(self.edge_idx), len(self.center_idx)) // max(
            self.n_edge_per, self.n_center_per)
        edge_pool = self.edge_idx.copy()
        center_pool = self.center_idx.copy()
        for _ in range(n_batches):
            if len(edge_pool) < self.n_edge_per:
                edge_pool = self.edge_idx.copy()
            if len(center_pool) < self.n_center_per:
                center_pool = self.center_idx.copy()
            self.rng.shuffle(edge_pool)
            self.rng.shuffle(center_pool)
            batch = np.concatenate([
                edge_pool[:self.n_edge_per],
                center_pool[:self.n_center_per],
            ])
            edge_pool = edge_pool[self.n_edge_per:]
            center_pool = center_pool[self.n_center_per:]
            yield batch

    def __len__(self):
        return max(len(self.edge_idx), len(self.center_idx)) // max(
            self.n_edge_per, self.n_center_per)
```

---

### 问题 11：LR scheduler 的 T_max 与实际步数不符

**位置**：`train_conv_spatial.py:180-181, 235, 338`

**现状**：
```python
scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs_sup + args.epochs_unsup)
```
但**有监督阶段提前 break**（line 264-266，RE<0.03）或 `--mode unsupervised` 时，
实际训练步数 ≠ `T_max`，**LR 余弦曲线会错位**（提前到达最低点或永远到不了）。

**修复方案**：两个阶段各用独立 scheduler：
```python
# 在阶段 1 前
scheduler_sup = CosineAnnealingLR(optimizer, T_max=args.epochs_sup)
# 阶段 1 用 scheduler_sup.step()

# 进入阶段 2 前重建 scheduler
scheduler_unsup = CosineAnnealingLR(optimizer, T_max=args.epochs_unsup)
# 阶段 2 用 scheduler_unsup.step()
```

---

### 问题 12：验证指标只有 RE，缺 SSIM/IoU

**位置**：`train_conv_spatial.py:237-256`

**现状**：只算 Relative Error (RE)。
**影响**：RE 是整体范数误差，对"内含物位置是否对、大小是否对"不敏感。
两个 RE 相同的预测，一个位置对了大小错了、一个位置错了大小对了，RE 可能相同。

**修复方案**：补充以下指标（验证阶段）：
```python
def compute_metrics(pred, gt, element_centers, sigma_threshold=0.025):
    """
    pred, gt: (B, n_elems)
    element_centers: (n_elems, 2) 用于算质心距离
    """
    metrics = {}

    # 1. RE（已有）
    metrics['RE'] = (torch.norm(pred - gt, dim=-1).mean() /
                     (torch.norm(gt, dim=-1).mean() + 1e-8)).item()

    # 2. CC：相关系数
    pred_c = pred - pred.mean(-1, keepdim=True)
    gt_c   = gt   - gt.mean(-1, keepdim=True)
    cc = (pred_c * gt_c).sum(-1) / (
        pred_c.norm(dim=-1) * gt_c.norm(dim=-1) + 1e-8)
    metrics['CC'] = cc.mean().item()

    # 3. Mask IoU：二值化后内含物位置重合度
    pred_mask = (pred > sigma_threshold)
    gt_mask   = (gt   > sigma_threshold)
    inter = (pred_mask & gt_mask).sum(-1).float()
    union = (pred_mask | gt_mask).sum(-1).float()
    metrics['IoU'] = (inter / (union + 1e-8)).mean().item()

    # 4. 位置误差：预测 vs GT 内含物质心距离 (cm)
    def centroid(mask):
        w = mask.float().unsqueeze(-1)                       # (B, N, 1)
        c = element_centers.unsqueeze(0).expand(pred.shape[0], -1, -1)
        return (c * w).sum(1) / (w.sum(1) + 1e-8)            # (B, 2)
    if pred_mask.any() and gt_mask.any():
        d = torch.norm(centroid(pred_mask) - centroid(gt_mask), dim=-1)
        metrics['PosErr_cm'] = (d.mean() * 100).item()  # m → cm

    return metrics
```

---

### 问题 13：缺少 EMA 模型权重

**现状**：无 EMA。高精度任务标配，通常能提升 1~3% RE。

**修复方案**：在 `train_conv_spatial.py` 加一个 EMA wrapper：
```python
class EMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model):
        for k, v in model.state_dict().items():
            if v.dtype.is_floating_point:
                self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1 - self.decay)
            else:
                self.shadow[k] = v.detach().clone()

    def apply_to(self, model):  # 验证/推理时调用
        model.load_state_dict(self.shadow)
```

训练循环中：
```python
ema = EMA(model)
# 每个 optimizer.step() 后
ema.update(model)

# 验证前
backup = {k: v.clone() for k, v in model.state_dict().items()}
ema.apply_to(model)
# ... 跑验证 ...
model.load_state_dict(backup)  # 恢复
```

---

### 问题 14：缺少 TTA（旋转对称增强）

**原理**：16 电极环具有旋转对称性。推理时把输入旋转 N 次（每次旋转电极对），
预测后反向旋转求平均，能显著降低噪声敏感度。

**修复方案**：在 `predict` 接口加 TTA：
```python
def predict_tta(self, voltages, n_rotations=8):
    """
    voltages: (n_freq, n_meas=208) 或 (B, n_freq, 208)
    相邻激励模式下，旋转 1 个电极 = 测量循环移位 13 个位置（每激励 13 测量）
    """
    preds = []
    n_meas = voltages.shape[-1]
    n_meas_per_exc = 13  # adjacent pattern
    for k in range(n_rotations):
        shift = k * n_meas_per_exc
        V_rot = torch.roll(voltages, shifts=shift, dims=-1)
        # 注意：σ 也需要对应旋转（按电极角度反算单元位置）
        sigma_rot = self.predict(V_rot)
        sigma_inv = rotate_sigma(sigma_rot, -k)  # 反向旋转回原坐标系
        preds.append(sigma_inv)
    return torch.stack(preds).mean(0)  # 平均
```

> `rotate_sigma` 需要根据单元中心坐标做坐标旋转 + 重采样，实现略复杂。
> 归入 P3 可选优化，优先级低于前面的问题。

---

## 五、P3 架构演进方向（可选）

> 📌 P3 都是"锦上添花"，P0-P2 落地后再考虑。方向 B/D 的收益直接受 Jacobian 精度制约。

### 方向 A：U-Net 风格的 Encoder-Decoder

**动机**：当前 `Conv → Sample → GNN` 缺少 decoder，**高分辨率空间信息无法带回来**。

**建议结构**：
```
Voltages → ConvEncoder(下采样) ─┬─ skip1 ──┐
                                ├─ skip2 ──┤
                                ↓          ↓
                           bottleneck → ConvDecoder(上采样)
                                              ↓
                              GridSample → GNN 精修 → σ
```

跳跃连接（skip）能大幅提升边缘锐度。GridSampler 在 decoder 高分辨率特征上采样，细节更丰富。

---

### 方向 B：Jacobian 先验注入（可选，收益受 Jacobian 精度制约）

> ⚠️ **评审降级**：原标"性价比最高"，但 P0-0 暴露出 Jacobian 在 σ>0.03 时精度不足。
> 用不准确的 Jacobian 做反投影初值，收益会打折。**仅在 Jacobian 可信度高时才推荐**。

**动机**：`Jacobian` 已经预计算（`data/generated/jacobian.npy`，shape `(6, 208, n_elems)`），
但**只在无监督 loss 里用，没作为模型输入**。

**建议**：把**反投影粗重建**作为先验输入给 decoder：
```python
# 粗重建初值: bp = J^T @ V_diff / scale
def backprojection_init(V_diff, jacobian):
    """
    V_diff: (B, n_meas)
    jacobian: (n_meas, n_elems)
    return: (B, n_elems) 物理合理的初值
    """
    return V_diff @ jacobian   # (B, n_elems)

# 在 ConvSpatialEIT.forward:
bp = backprojection_init(voltages_diff, self.jacobian)  # (B, n_elems)
bp_feat = bp.unsqueeze(-1)  # (B, n_elems, 1)
node_feat = torch.cat([node_feat, bp_feat, pos_encoding], dim=-1)
```

网络只需做 **"refinement"**（残差修正），收敛更快、精度更高。
> 注意：本场景 Jacobian 在高对比度下偏差大，建议配合方向 C（重线性化）或谨慎使用。

---

### 方向 C：可微 FEM / 迭代重线性化（彻底解决梯度偏差）

> 🆕 **新增**：这是 P0-0（MCL 梯度偏差）的根治方案。

**动机**：P0-0 只把 MCL 从纯 Jacobian 改成 full_fem，但**梯度回传仍走 Jacobian**（`loss.py:117-122`）。
要彻底消除偏差，需要让 FEM 正解本身可微分，或定期在当前 σ_pred 处**重新计算 Jacobian**。

**方案 C1 — 迭代重线性化（中等成本）**：
每隔 N 步，在当前 `σ_pred` 处重新计算 Jacobian（而非固定用 σ_ref=0.01 处的）：
```python
# 每 N 步重算 Jacobian（基于当前模型预测的 σ 分布）
if step % recompute_interval == 0:
    with torch.no_grad():
        # 用当前 σ_pred 作为新的线性化点
        new_J = solver.compute_jac_at(sigma_pred.detach())
    mcl.jacobian = new_J
```
pyEIT 支持任意 σ 处的 Jacobian 计算（`_forward_op.compute_jac(sigma)`），
只需封装一个接口。成本：每 N 步多算一次 Jacobian（O(n_meas×n_elems)）。

**方案 C2 — 可微 FEM（高成本，最终方案）**：
把 pyEIT 的 FEM 正解用 PyTorch 重写（或用 [dolfin-adjoint](https://dolfin-adjoint.org/) 这类自动微分工具），
让 `V = F(σ)` 整个可微，梯度**直接通过 FEM 方程回传**，完全消除线性近似误差。
这是学术界 EIT 深度学习的终极方案，但实现工作量大（1-2 周）。

> **建议**：先用 C1（迭代重线性化）验证收益，若 Jacobian 偏差确实是瓶颈，再投入 C2。

---

## 六、实施路线图

> 📌 **顺序原则（融合评审意见）**：先修 bug → 再修架构 → 最后修泛化。
> 每阶段独立可验证，前一阶段不依赖后一阶段。

### 第一阶段（半天）—— 修 bug + 解锁架构瓶颈

| # | 任务 | 文件 | 预期 |
|---|---|---|---|
| P0-0 | **MCL 强制 full_fem 模式**（修梯度 bug） | `train_conv_spatial.py:295` | 无监督精度天花板抬高 |
| P0-1 | 多频率退化为单频 | `models/conv_spatial_eit.py` | 释放模型容量 |
| P0-2 | 位置编码注入 GNN | `models/conv_spatial_eit.py` | 中心区域重建清晰 |
| P1-6 | 删除 SmoothnessLoss | `training/loss.py`, `train_conv_spatial.py` | 去除错误正则 |
| P1-7 | 有监督阶段关闭 SigmaDeviationLoss | `train_conv_spatial.py` | 停止对抗监督 |

**预期 RE**：0.088 → **0.06** 左右（消除 bug + 释放容量 + 位置信息）

**验收**：
- Conv encoder 输入通道 = 1
- `pos_encoding` buffer 注册成功，GNN 第一层维度更新
- 有监督阶段 loss 只剩 MSE
- 无监督阶段 loss 输出不再异常

---

### 第二阶段（半天）—— 修正错误正则 + 输入处理

| # | 任务 | 文件 | 预期 |
|---|---|---|---|
| P1-4 | GridSampler 保持 13×16 不插值，删下采样 | `models/conv_spatial_eit.py` | 空间细节保留 |
| P1-5 | TV 改用真实 FEM 邻接 | `training/loss.py`, `train_conv_spatial.py` | TV 物理正确 |
| P1-8 | 输入电压归一化 | `models/conv_spatial_eit.py` | 训练更稳 |
| P1-7b | 无监督阶段 SigmaDeviationLoss 降权/软化 | `training/loss.py` | 避免对抗监督 |

**预期 RE**：0.06 → **0.05~0.06**（架构 + 正则修正）
> ⚠️ 可能达不到评审预估的 0.04-0.05，因为 Jacobian 梯度偏差（P0-0 残余）仍是天花板。

**验收**：
- encoder 输出 shape 为 `(B, 128, 13, 16)`
- TV 的 `edge_idx` 与模型 `setup_mesh` 一致
- 验证集 RE 有可见下降

---

### 第三阶段（1 天）—— 域随机化（通用化的关键）

| # | 任务 | 文件 | 预期 |
|---|---|---|---|
| P1-9 | 域随机化（电导率浮动 + 噪声扩范围 + 增益扰动 + 接触阻抗 + 基线漂移） | `data/generate_circle_dataset.py` | **泛化能力提升（仿真 RE 可能略升，真实数据大幅改善）** |

**关键认知**：这一步在仿真集上的 RE **可能略升**（因为训练数据变难了），
但模型对真实测量的泛化能力会**大幅改善**——这才是"通用 EIT"的真正目标。
**不要用仿真集 RE 是否下降来判断这一步是否成功**，要看在分布外（OOD）测试集的表现。

**验收**：
- 重新生成数据集，样本电导率/噪声/增益都有随机变化
- 训练后在原验证集（理想条件）RE 保持，在加噪/扰动验证集上 RE 退化更小

---

### 第四阶段（1 天，可选）—— 训练流程打磨

| # | 任务 | 文件 | 预期 |
|---|---|---|---|
| P2-10 | 修复 BatchSampler | `train_conv_spatial.py` | 训练稳定性 |
| P2-11 | 双阶段独立 scheduler | `train_conv_spatial.py` | LR 曲线正确 |
| P2-12 | 验证指标补充 CC/IoU/PosErr | `train_conv_spatial.py` | 评估更全面 |
| P2-13 | EMA 权重 | `train_conv_spatial.py` | +1~3% RE |

**预期 RE**：再降 0.01~0.02（流程优化累积效应）

---

### 第五阶段（可选，P3）—— 架构演进

| # | 任务 | 文件 | 预期 |
|---|---|---|---|
| P3-方向B | Jacobian 先验注入 | `models/conv_spatial_eit.py` | **收益取决于 Jacobian 精度**（高对比度下打折） |
| P3-方向A | U-Net Encoder-Decoder | `models/conv_spatial_eit.py` | 边缘锐度提升 |
| P3-方向C1 | 迭代重线性化 Jacobian | `training/loss.py` | 彻底解决 P0-0 残余偏差 |
| P3-方向C2 | 可微 FEM | — | 终极方案，工作量 1-2 周 |
| P3-14 | TTA 旋转增强 | `models/conv_spatial_eit.py` | 降噪声敏感度 |

> 建议顺序：C1（验证 Jacobian 偏差是否真是瓶颈）→ B（用准确 Jacobian 做先验）→ A → C2

---

## 附录：快速验证清单

每完成一项修改后，建议跑：
```bash
# 1. 网络前向测试（确认 shape 正确）
python models/conv_spatial_eit.py

# 2. 短训练验证（10 epoch）
python train_conv_spatial.py --epochs_sup 10 --batch_size 32

# 3. 检查验证指标是否合理（RE 应该 < 0.1，CC > 0.5）
```

**目标基线**（单圆数据集，理想仿真条件）：
- RE：当前 ~0.088 → 目标 **0.05~0.06**（⚠️ 因 Jacobian 梯度偏差是天花板，0.04 可能达不到，除非做 P3-方向C）
- CC > 0.85
- IoU > 0.7
- PosErr < 1cm

> ⚠️ **域随机化后**（第三阶段），理想仿真条件下的 RE 可能略升（数据变难），
> 但 OOD/真实条件下的 RE 会大幅改善。**通用化的衡量应以 OOD 表现为准，而非单一仿真集 RE。**

---

## 附录：参考文献与思路来源

- **Deep D-bar / D-bar + CNN**：EIT 经典物理引导重建
- **Jacobian / 反投影先验注入**：Seo et al., EIT deep learning 综述
- **位置编码 + GNN**：借鉴 GraphTransformer / NeRF 位置编码
- **EMA + TTA**：图像分类/医学影像通用提精度技巧
- **TV on FEM mesh**：EIT 经典正则（不是像素域 TV）

---

---

## 附录：版本变更记录

- **v1.1（2026-06-15）**：采纳评审意见后的重大修订
  - 🔴 新增 P0-0：MCL 走 Jacobian 模式导致梯度偏差（实际 bug）
  - 🟠 Conv2D 错配降级：删 B/C 重构方案，只保留输入归一化（移到 P1-8）
  - 🟠 GridSampler 方案修订：从"插值到 16×16"改为"保持 13×16 原生不插值"
  - 🟠 域随机化从 P3 升到 P1-9（sim-to-real 是通用化真正的天花板）
  - 🟡 Jacobian 先验注入降为 P3 可选（受 Jacobian 精度制约）
  - 🆕 新增 P3-方向C：迭代重线性化 / 可微 FEM（彻底解决梯度偏差）
  - 📅 实施路线图重排：修 bug → 修架构 → 修泛化 → 打磨 → 演进
  - 📊 RE 预期数字修正：0.04-0.05 改为更现实的 0.05-0.06（标注天花板原因）

- **v1.0（2026-06-15）**：初版，14 个优化点全梳理

---

*文档版本：v1.1  |  维护者：ZCode  |  最后更新：2026-06-15*
