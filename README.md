# EIT 植物根部无监督成像

基于 pyEIT + PyTorch，无监督深度学习桶式植物根部EIT成像系统。

**硬件场景**：圆柱形桶（直径≈20cm），单环16电极，2D截面采集。

---

## 🚀 快速开始

### 1. 安装依赖

```bash
cd eit_root_imaging
pip install -r requirements.txt
```

### 2. 首次训练（自动生成数据 → 自动训练）

```bash
python train.py
```

数据只生成一次，后续训练直接复用：

```bash
# 第二次以后只需：
python train.py
python train.py --epochs 300
python train.py --batch_size 64
```

### 3. 强制重新生成数据

```bash
python train.py --generate
```

### 4. 恢复中断的训练

```bash
python train.py --resume checkpoints/eit_20260610/checkpoint_epoch_50.pt
```

---

## 🔄 工作流说明

```
第一次运行:
  train.py
    ├─ 检测到 data/generated/eit_dataset.h5 不存在
    ├─ 自动调用 generate_dataset.py 生成数据  ← 较慢，仅一次
    ├─ 自动调用 precompute_jacobian.py 计算    ← 较慢，仅一次
    └─ 启动无监督训练

第二次及以后:
  train.py
    ├─ 检测到数据已存在 → 跳过
    └─ 直接启动训练
```

> ⚡ **数据只需生成一次**。`generate_dataset.py` 生成 10,000+500+200 样本约需几分钟到十几分钟（取决于网格分辨率），生成后所有后续训练直接读取 HDF5 文件。

---

## 📋 文件说明

| 文件 | 功能 | 运行方式 |
|------|------|----------|
| `train.py` | **统一训练入口** | `python train.py` |
| `config/mesh_config.yaml` | 网格与电极配置 | 被引用 |
| `config/train_config.yaml` | 训练超参数配置 | 被引用 |
| `data/generate_dataset.py` | 生成 HDF5 数据集 | 自动调用 |
| `data/precompute_jacobian.py` | 预计算雅可比矩阵 | 自动调用 |
| `training/unsupervised_loop.py` | 无监督训练核心 | 被 `train.py` 调用 |
| `evaluation/evaluate.py` | 测试集评估 | `python evaluation/evaluate.py --checkpoint ...` |

---

## 🎯 三种运行模式

```bash
# 模式1: 标准训练（推荐）
python train.py

# 模式2: 调参
python train.py --epochs 500 --batch_size 64 --n_train 20000

# 模式3: 恢复/微调
python train.py --resume checkpoints/xxx.pt --epochs 100
```
