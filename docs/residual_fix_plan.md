# 残差网络效果差的修复计划

## 问题诊断结果

### 关键问题

1. **delta_scale 过小 (0.02)**: 限制了模型的最大校正能力，25%的单元需要更大的校正
2. **传统重建质量极差**: RE(sigma_0)=1.48，BP方法几乎完全失败
3. **验证指标不变**: 模型无法在±0.02范围内做出有意义的改进

## 修改方案

### Phase 1: 修改配置和模型代码

#### 1.1 增大 delta_scale

修改 `config/residual_eit_config.yaml`:
```yaml
model:
  delta_scale: 0.10  # 从 0.02 改为 0.10，允许更大的校正
```

#### 1.2 移除过于严格的 tanh 约束（可选）

修改 `models/residual_mesh_gnn.py`:
```python
# 方案A: 保持tanh但增大delta_scale
return self.delta_scale * torch.tanh(raw_delta)

# 方案B: 使用softplus替代tanh（更平滑）
return self.delta_scale * F.softplus(raw_delta) - self.delta_scale * 0.5

# 方案C: 完全移除激活函数，靠clamp限制
return raw_delta  # 外层sigma计算时已有clamp
```

推荐使用方案A（增大delta_scale），最简单且可控。

#### 1.3 增加sigma_min/sigma_max的范围检查

确保 `sigma_min` 和 `sigma_max` 配置合理：
```yaml
model:
  sigma_min: 0.005
  sigma_max: 0.15  # 从0.1增加到0.15，允许更高的电导率
```

### Phase 2: 改进传统重建方法

#### 2.1 使用 JAC 方法重新预计算残差特征

```bash
python data/precompute_residual_features.py \
  --h5 data/generated/mixed_dataset.h5 \
  --method jac \
  --force
```

JAC 方法比 BP 更稳定，预期 RE(sigma_0) 从 1.48 降低到 ~0.8。

#### 2.2 如果 JAC 也失败，考虑 Tikhonov 正则化

在 `models/traditional/reconstructor.py` 中添加 Tikhonov 方法：
```python
class TikhonovReconstructor(BaseReconstructor):
    def reconstruct(self, voltage):
        # sigma_0 = sigma_ref + J.T @ inv(J @ J.T + lambda*I) @ r
        ...
```

### Phase 3: 训练策略优化

#### 3.1 调整损失权重

```yaml
training:
  loss_weights:
    supervised: 1.0
    residual_measurement: 0.5  # 降低权重，避免与监督损失冲突
    tv: 0.02
    delta_l1: 0.005  # 降低稀疏约束
    delta_smooth: 0.01
```

#### 3.2 增加学习率和batch size

```yaml
training:
  batch_size: 16  # 从8增加到16
  learning_rate: 5.0e-4  # 从3e-4增加到5e-4
```

## 实施顺序

1. **立即修改**: 
   - config/residual_eit_config.yaml (delta_scale: 0.02 → 0.10)
   
2. **重新预计算**: 
   - 用 JAC 方法重新计算 sigma_0, g, residual

3. **重新训练**:
   - 运行 train_residual_eit.py

4. **观察结果**:
   - 如果 RE(sigma_0) < 1.0，继续训练
   - 如果仍然很高，考虑实现 Tikhonov

5. **进一步优化**:
   - 如果效果好，尝试更大的 delta_scale (0.15)
   - 调整损失权重

## 预期效果

| 修改 | 预期 RE(sigma_0) | 预期最终 RE |
|------|------------------|-------------|
| 当前 (BP + delta=0.02) | 1.48 | 2.10 (不变) |
| delta=0.10 | 1.48 | ~1.0 |
| JAC + delta=0.10 | ~0.8 | ~0.4-0.5 |
| JAC + delta=0.15 | ~0.8 | ~0.3-0.4 |

## 文件修改清单

1. `config/residual_eit_config.yaml` - 修改 delta_scale, sigma_max, loss_weights
2. `data/precompute_residual_features.py` - 确保 JAC 方法正确实现
3. `models/residual_mesh_gnn.py` - 可选：修改激活函数
4. `training/residual_trainer.py` - 可选：添加更详细的日志

## 验证脚本

训练完成后运行：
```bash
python diagnose_residual.py
```

检查：
- RE(sigma_0) 是否降低
- delta_sigma 范围是否合理
- 最终 RE 是否有改善