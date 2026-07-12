# 无监督训练诊断报告

## 问题概述

无监督训练在 Epoch 30 左右发生严重崩溃，Loss 从 ~0.16 暴涨到 ~0.40，
远比有监督预训练结束时的 RE=0.182 更差。

---

## ✅ 已修复 (2026-06-23)

### 修复内容

| 问题 | 修复方案 |
|------|----------|
| **学习率重启** | 移除 `CosineAnnealingWarmRestarts`，改用稳定的 `CosineAnnealingLR` |
| **学习率过高** | 无监督阶段使用独立优化器，默认 `lr=1e-5`（比预训练小10倍） |
| **缺少验证监控** | 添加每 epoch 验证 RE 计算，实时监控模型状态 |
| **自适应权重失衡** | 移除 `AdaptiveLossWeighter`，改用固定权重 |
| **缺少 Early Stopping** | 添加 patience=10 的早停机制 |
| **梯度裁剪过松** | 从 5.0 改为 1.0 |
| **FEM 求解频率低** | 从 20 步改为 10 步 |

### 新增命令行参数

```bash
--unsup_lr 1e-5           # 无监督学习率（默认1e-5）
--early_stop_patience 10  # 早停耐心值（默认10，0=关闭）
```

### 修复后的训练流程

```
阶段1: 有监督预训练
  - 学习率: 1e-4
  - 调度器: CosineAnnealingLR (无重启)
  - Epoch 50 → RE ≈ 0.18

阶段2: 无监督精调
  - 学习率: 1e-5 (独立优化器)
  - 调度器: CosineAnnealingLR (无重启)
  - 固定损失权重: sup=0.5, m=1.0, t=0.05, d=0.1
  - 每 epoch 验证监控
  - Early Stopping (patience=10)
```

---

## 关键发现

### 1. 训练崩溃时间点

```
Epoch 1-29:  Loss 0.30 → 0.16（正常下降）
Epoch 30:    Loss 跳到 0.22（开始崩溃）
Epoch 31-50: Loss 0.37-0.40（彻底失控）
```

### 2. 根本原因分析

#### A. 学习率调度器问题（主因）

`train_conv_spatial.py` 第300-302行使用 CosineAnnealingWarmRestarts：
```python
scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=2)
```

这意味着每20个epoch学习率**强制重启到初始值**！
- Epoch 20：第一次重启 → 学习率突然变大 → 模型震荡
- Epoch 40：第二次重启 → 学习率又变大 → 再次震荡

崩溃发生在 Epoch 30（介于20和40之间），说明：
- Epoch 20 重启后模型开始不稳定
- Epoch 30 后彻底崩溃
- Epoch 40 重启进一步加剧

**问题**：WarmRestarts 设计用于长训练（几百epoch），不适合50 epoch的无监督精调。

#### B. 无监督循环缺少验证监控

`train_conv_spatial.py` 第512-584行，无监督训练只记录训练loss：
```python
print(f"  Unsup Epoch {epoch:2d} | Loss: {epoch_loss/len(train_loader):.4f}")
recorder.log_epoch(phase="unsupervised", epoch=epoch, loss=epoch_loss/len(train_loader))
```

**没有计算验证集 RE/CC**！无法及时发现模型正在恶化。

#### C. AdaptiveLossWeighter 设计缺陷

`loss.py` 第343-368行：
```python
precision = torch.exp(-self.log_vars[i])
total += precision * loss + self.log_vars[i] * 0.5 * loss_scale
```

问题：
- `precision = exp(-log_var)` → log_var越大，权重越小
- 同时惩罚项 `log_var * 0.5 * loss_scale` → 鼓励log_var变大
- 可能导致某些损失项权重失控衰减，物理约束失效

#### D. 半监督混合策略问题

第532-540行：
```python
loss_sup = criterion(sp, S_gt)  # 使用 GT 作为锚点
loss_dict = {"loss_sup": loss_sup, "loss_m": loss_m, ...}
```

这不是纯无监督，而是半监督。但：
- 自适应权重可能让 `loss_m`（物理约束）权重过低
- 模型可能主要靠 `loss_sup` 学习，物理约束未起作用

### 3. 其他观察

#### 有监督预训练效果良好
- 50 epoch 后 RE = 0.182（相对误差18.2%）
- 这已经是可接受的结果

#### Jacobian 线性近似的有效范围
- Jacobian 在 σ_ref = 0.01 附近有效
- 如果 σ_pred 偏离太远，线性近似失真
- `SigmaDeviationLoss` 试图约束这点，但可能权重不足

## 解决方案

### 立即可行的修复

1. **改用稳定的学习率调度**
   ```python
   # 替换 WarmRestarts
   scheduler = CosineAnnealingLR(optimizer, T_max=epochs_unsup, eta_min=1e-6)
   ```

2. **添加验证监控**
   ```python
   # 在无监督循环中添加
   if epoch % 5 == 0:
       val_re = validate(model, val_loader)
       if val_re > best_re * 1.1:  # 如果恶化超过10%
           print("⚠ 模型恶化，提前停止")
           break
   ```

3. **固定损失权重（代替自适应）**
   ```python
   weights = {
       "loss_sup": 0.5,   # 有监督锚点
       "loss_m": 1.0,     # 物理约束（核心）
       "loss_t": 0.05,    # TV
       "loss_d": 0.1,     # 偏离约束
   }
   total = sum(weights[k] * v for k, v in loss_dict.items())
   ```

4. **减小学习率**
   ```python
   # 无监督精调用更小的学习率
   lr = 1e-5  # 比预训练的1e-4小10倍
   ```

### 进阶改进

1. **纯无监督模式**：移除 `loss_sup`，完全依赖物理约束
2. **更频繁的 FEM 求解**：`fem_interval=5` 而非20
3. **梯度裁剪**：减小到 1.0 而非 5.0
4. **Early Stopping**：监控验证 RE，恶化时停止

## 结论

无监督训练失败的主要原因是：
1. **学习率调度器重启导致模型震荡**
2. **缺少验证监控无法及时发现恶化**
3. **自适应损失权重可能导致物理约束失效**

修复后预期效果：无监督精调应能维持或略微改善有监督预训练的 RE=0.182。