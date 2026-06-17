# Conv-Spatial EIT 优化实施设计文档 (SDD)

> 基于 [网络瓶颈分析](./network_bottleneck_analysis.md) 的具体实施步骤
> 基准模型: ConvSpatialEIT v2, gnn_hidden=512, 4.2M params, RE=0.108

---

## Phase 1: P0 低成本高收益修复 (预计 1-2h)

### 1.1 6 频浅层融合 — 有监督 RE 0.108 → 0.07

#### 目标
当前模型丢弃 5/6 的频率信息。用 1 层 Conv 将 6 频融合为 1 频，仅增 6 个参数。

#### 实施

**文件**: `models/conv_spatial_eit.py`

**1) 在 `__init__` 中添加频率融合层** (第 180 行后):

```python
# 0. 多频融合层（6频 → 1频）
self.freq_fusion = nn.Conv2d(in_channels=n_frequencies, out_channels=1,
                              kernel_size=1, bias=False)
```

**2) 修改 `forward`** (第 294-299 行):

```python
# 修改前:
if voltages.dim() == 3:
    x = voltages[:, :1, :].view(B, 1, 13, 16)
else:
    x = voltages[:, :1, :, :]

# 修改后:
if voltages.dim() == 3:
    x = voltages.view(B, 6, 13, 16)
else:
    x = voltages
x = self.freq_fusion(x)  # (B, 1, 13, 16) — 自动学习权重
```

**参数量变化**: +6 (可忽略)

#### 验证

```bash
# 快速测试：加载旧权重，验证不崩溃
python -c "
from models.conv_spatial_eit import ConvSpatialEIT
from data.eit_forward import EITForwardSolver
import torch
solver = EITForwardSolver('config/mesh_config.yaml')
m = ConvSpatialEIT(gnn_hidden=512, gnn_layers=4)
m.setup_mesh(solver.element_centers[:,:2], solver.mesh.element)
x = torch.randn(2, 6, 13, 16)
out = m(x)
print(f'Output: {out[\"sigma\"].shape}, range: [{out[\"sigma\"].min():.4f}, {out[\"sigma\"].max():.4f}]')
"
```

```bash
# 完整训练验证
python train_conv_spatial.py --mode supervised --epochs_sup 30 --hidden_dim 512 --batch_size 32
# 预期: RE < 0.08
```

---

### 1.2 半监督混合损失 — 无监督 RE 0.54 → 0.12

#### 目标
无监督精调不退化，通过保留有监督锚点信号。

#### 实施

**文件**: `train_conv_spatial.py`

**1) 修改无监督损失** (约第 320 行):

```python
# 修改前:
total = loss_m + 0.05 * loss_t + 0.01 * loss_d

# 修改后:
# 半监督: 有监督 MSE + 物理约束
loss_sup = criterion(out['sigma'], batch['sigmas'].to(device))
total = 0.3 * loss_sup + 0.7 * (loss_m + 0.1 * loss_t + 0.01 * loss_d)
```

**2) 加载 sigma GT** (约第 309 行):

```python
# 修改前:
V = batch['voltages'].to(device).view(-1, 6, 13, 16)

# 修改后:
V = batch['voltages'].to(device).view(-1, 6, 13, 16)
S = batch['sigmas'].to(device)  # 需要 GT 做半监督锚点
```

**3) 添加平滑损失** (新增):

```python
# 在文件顶部导入
from training.loss import SmoothnessLoss
sml = SmoothnessLoss(
    element_centers=torch.from_numpy(centers).float(),
    mesh_elements=torch.from_numpy(elements).long(),
    mesh_nodes=torch.from_numpy(solver.mesh.node[:, :2]).float(),
)
```

```python
# 损失组合:
loss_smooth = sml(sp)
loss_sup = criterion(sp, S)
total = 0.3 * loss_sup + 0.5 * loss_m + 0.1 * loss_t + 0.05 * loss_smooth + 0.05 * loss_d
```

#### 验证

```bash
python train_conv_spatial.py --mode both --epochs_sup 30 --epochs_unsup 30 --hidden_dim 512 --batch_size 32
# 预期: 无监督后 RE 0.10–0.15 (不退化)
```

---

### 1.3 无监督阶段也保存最佳模型

**文件**: `train_conv_spatial.py` 无监督循环 (~第 335 行后)

```python
# 在 unsup epoch 循环中添加:
val_re = compute_re_on_validation(model, val_loader, device)
if val_re < best_unsup_re:
    best_unsup_re = val_re
    torch.save(model.state_dict(), "checkpoints/conv_spatial_unsup_best.pt")
    recorder.log_event("best_unsup_saved", re=val_re.item())
```

---

## Phase 2: P1 中等复杂度改进 (预计 3-4h)

### 2.1 多样化数据生成器

#### 目标
从"单圆"扩展为椭圆、多圆、不规则形状，提升泛化。

#### 实施

**文件**: `data/generate_circle_dataset.py` → 重命名为 `data/generate_mixed_dataset.py`

**1) 形状类型枚举**:

```python
class RootShapeType:
    CIRCLE = 0      # 单圆 (原)
    ELLIPSE = 1     # 椭圆
    MULTI_CIRCLE = 2 # 多圆 (2-5个)
    IRREGULAR = 3   # 不规则多边形
```

**2) 每个形状生成器**:

```python
def generate_ellipse_sigma(centers, n_elems, soil_sigma=0.01, root_sigma=0.05):
    """椭圆内含物"""
    cx, cy = random_pos()          # 随机中心
    a = random.uniform(0.02, 0.06) # 半长轴
    b = random.uniform(0.01, 0.04) # 半短轴
    angle = random.uniform(0, np.pi)
    # 旋转后的椭圆距离
    dx = centers[:,0] - cx
    dy = centers[:,1] - cy
    rx = dx * np.cos(angle) + dy * np.sin(angle)
    ry = -dx * np.sin(angle) + dy * np.cos(angle)
    mask = (rx/a)**2 + (ry/b)**2 <= 1.0
    sigma = np.full(n_elems, soil_sigma)
    sigma[mask] = root_sigma
    return sigma

def generate_multi_circle_sigma(centers, n_elems, ...):
    """2-5 个随机圆"""
    n = random.randint(2, 5)
    mask = np.zeros(n_elems, dtype=bool)
    for _ in range(n):
        cx, cy = random_pos()
        r = random.uniform(0.01, 0.04)
        mask |= (centers[:,0]-cx)**2 + (centers[:,1]-cy)**2 <= r**2
    sigma = np.full(n_elems, soil_sigma)
    sigma[mask] = root_sigma
    return sigma

def generate_irregular_sigma(centers, n_elems, ...):
    """Bezier 曲线控制点生成不规则形状"""
    n_ctrl = random.randint(4, 8)
    angles = np.sort(np.random.uniform(0, 2*np.pi, n_ctrl))
    radii = np.random.uniform(0.015, 0.05, n_ctrl)
    cx, cy = random_pos()
    # Bezier 插值生成边界
    ...
```

**3) 统一生成入口**:

```python
def generate_sample(centers, n_elems):
    shape_type = random.choice(list(RootShapeType))
    contrast = random.uniform(3, 20)  # 可变对比度
    root_sigma = soil_sigma * contrast
    if shape_type == RootShapeType.CIRCLE:
        return generate_circle_sigma(...)
    elif shape_type == RootShapeType.ELLIPSE:
        return generate_ellipse_sigma(...)
    elif ...:
        ...
```

**4) 数据集混合比例**:

```python
# 默认比例
SHAPE_DISTRIBUTION = {
    RootShapeType.CIRCLE:    0.25,  # 25% 单圆
    RootShapeType.ELLIPSE:   0.30,  # 30% 椭圆
    RootShapeType.MULTI_CIRCLE: 0.30, # 30% 多圆
    RootShapeType.IRREGULAR: 0.15,  # 15% 不规则
}
```

#### 验证

```bash
python data/generate_mixed_dataset.py --n_train 20000 --n_val 500 --n_test 500
python train_conv_spatial.py --mode supervised --epochs_sup 50 --hidden_dim 512
# 预期: mixed 数据的 RE ~0.10–0.12 (略高于单圆因为形状更难)
# 同时在单圆集和根系集上评估，验证泛化
```

---

### 2.2 GNN GAT 注意力聚合

#### 目标
将 `SimpleGNNLayer` 的静态 sum 聚合替换为可学习的注意力聚合。

#### 实施

**文件**: `models/conv_spatial_eit.py`

**1) 新增 `GATLayer` 类** (~第 119 行):

```python
class GATLayer(nn.Module):
    """GAT-style 图注意力层（内存高效稀疏实现）"""
    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.1, heads: int = 4):
        super().__init__()
        self.heads = heads
        self.out_dim = out_dim
        assert out_dim % heads == 0
        self.head_dim = out_dim // heads

        # 注意力投影
        self.W_src = nn.Linear(in_dim, out_dim, bias=False)
        self.W_dst = nn.Linear(in_dim, out_dim, bias=False)
        self.attn = nn.Linear(out_dim * 2, heads, bias=False)
        self.dropout = nn.Dropout(dropout)

        self.mlp = nn.Sequential(
            nn.Linear(out_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(out_dim, out_dim),
            nn.LayerNorm(out_dim),
        )

    def forward(self, x, edge_idx, edge_weight):
        B, N, D = x.shape
        device = x.device
        src, dst = edge_idx
        n_edges = edge_idx.shape[1]

        h_src = self.W_src(x).view(B, N, self.heads, self.head_dim).permute(0,2,1,3)
        h_dst = self.W_dst(x).view(B, N, self.heads, self.head_dim).permute(0,2,1,3)

        # 稀疏注意力（chunk 处理避免 OOM）
        h_agg = torch.zeros(B, self.heads, N, self.head_dim, device=device)
        chunk = 20000
        for start in range(0, n_edges, chunk):
            end = min(start + chunk, n_edges)
            s, d = src[start:end], dst[start:end]
            ww = edge_weight[start:end]

            a_input = torch.cat([h_src[:, :, s], h_dst[:, :, d]], dim=-1)
            alpha = self.attn(a_input)  # (B, heads, chunk, heads)
            alpha = torch.softmax(alpha * (ww.view(1, 1, -1, 1) + 1e-8), dim=2)
            alpha = self.dropout(alpha)

            msg = h_src[:, :, s] * alpha  # weighted message
            h_agg.scatter_add_(2, d.view(1, 1, -1, 1).expand(B, self.heads, -1, self.head_dim), msg)

        h_agg = h_agg.permute(0, 2, 1, 3).reshape(B, N, self.out_dim) / (self.heads ** 0.5)

        h = torch.cat([x, h_agg], dim=-1) if D < self.out_dim else h_agg + x[:, :, :self.out_dim]
        h = self.mlp(h.reshape(B * N, -1)).reshape(B, N, -1)
        return h
```

**2) `ConvSpatialEIT.__init__` 中支持选择 GNN 类型** (第 188 行):

```python
self.gnn_type = gnn_type  # 'simple' | 'gat'
```

**3) 修改 `setup_mesh` 中 GNN 构建** (第 267-275 行):

```python
LayerClass = GATLayer if self.gnn_type == 'gat' else SimpleGNNLayer
self.gnn_blocks = nn.ModuleList([
    LayerClass(
        gnn_in_dim if i == 0 else self.gnn_hidden,
        self.gnn_hidden,
        self.gnn_dropout,
    )
    for i in range(self.gnn_layers)
])
```

#### 验证

```bash
python train_conv_spatial.py --mode supervised --epochs_sup 30 --hidden_dim 512 --gnn_type gat --batch_size 16
# GAT 单层有 4 个 attention heads，显存占用更大，降 batch_size
# 预期: RE < 0.095
```

---

## Phase 3: P2 精度提升 (预计 2-3h)

### 3.1 特征图上采样

#### 目标
Grid 分辨率从 13×16 (208 点) 提升到 26×32 (832 点)。

#### 实施

**文件**: `models/conv_spatial_eit.py`

**1) 在 `ConvSpatialEIT.__init__` 中添加上采样层**:

```python
self.upsample = nn.Sequential(
    nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2, bias=False),
    nn.BatchNorm2d(64),
    nn.ReLU(inplace=True),
    nn.Conv2d(64, 128, 3, padding=1, bias=False),
    nn.BatchNorm2d(128),
    nn.ReLU(inplace=True),
)
```

**2) 修改 forward 在 GridSampler 之前** (第 306-309 行):

```python
feat = self.encoder(x)      # (B, 128, 13, 16)
feat = self.upsample(feat)  # (B, 128, 26, 32)  ← 新增
node_feat = self.sampler(feat)
```

#### 验证

```bash
python train_conv_spatial.py --mode supervised --epochs_sup 30 --hidden_dim 512 --batch_size 24
# GridSampler 坐标是新分辨率，自动适应
# 预期: RE 0.108 → 0.08–0.09
```

---

### 3.2 可学习位置编码

#### 目标
固定 Fourier 编码 → 可学习 MLP 编码。

#### 实施

**文件**: `models/conv_spatial_eit.py`

**1) 替换位置编码生成** (~第 258 行):

```python
# 修改前:
pe = torch.cat([
    torch.from_numpy(pos).float(),
    fourier(pos),
    torch.from_numpy(radius).float(),
], dim=-1)

# 修改后:
self.learnable_pe = nn.Sequential(
    nn.Linear(3, 32),   # (x, y, r) 原始坐标
    nn.GELU(),
    nn.Linear(32, 32),
)
raw = torch.from_numpy(np.concatenate([pos, radius], axis=-1)).float()
pe = self.learnable_pe(raw)  # (n_elems, 32)
```

**2) 更新 pos_dim**:

```python
self.pos_dim = 32  # 固定 32 维
```

#### 验证

```bash
python train_conv_spatial.py --mode supervised --epochs_sup 30 --hidden_dim 512
# 预期: RE 略降 5-10%，位置编码适应网格结构
```

---

## 实施依赖关系

```
Phase 1 (独立，可并行)
├── 1.1 6频融合 ─────────── 无依赖
├── 1.2 半监督混合损失 ──── 无依赖
└── 1.3 无监督最佳保存 ──── 无依赖

Phase 2 (建议 2.1 先于 2.2)
├── 2.1 数据多样化 ──────── 依赖: 1.1 (用新数据训新模型)
└── 2.2 GAT 注意力 ──────── 可独立于 2.1

Phase 3 (建议 3.1 先于 3.2)
├── 3.1 上采样 ──────────── 依赖: 1.1 (可选叠加)
└── 3.2 可学习位置编码 ──── 依赖: 无
```

## 回滚策略

每个 Phase 独立，修改前创建分支:

```bash
git checkout -b phase-1-fusion
# ... 实施 1.1 + 1.2 ...
git checkout -b phase-2-data
# ... 实施 2.1 ...
```

任一 Phase 效果不佳时，可独立回滚，不影响其他优化。

## 测试 Checklist

| Phase | 测试项 | 验收标准 |
|:-----:|--------|----------|
| 1.1 | 前向传播 | 不崩溃，sigma 范围正常 |
| 1.1 | 有监督 30 epoch | RE < 0.08 |
| 1.2 | 无监督不退化 | RE < 0.15 |
| 1.2 | 无监督 RE 趋势 | 持续下降或稳定 |
| 2.1 | 数据生成 | 生成 20K 样本 < 5min |
| 2.1 | 形状分布 | 每种形状都有合理样例 |
| 2.1 | 跨数据评估 | 单圆/混合/根系 RE 差异 < 0.05 |
| 2.2 | 显存 | batch_size=16 不 OOM |
| 2.2 | RE 提升 | 相比简单 GNN 降低 ≥ 10% |
| 3.1 | 推理正确 | GridSampler 坐标匹配 |
| 3.1 | 显存 | batch_size=24 不 OOM |
