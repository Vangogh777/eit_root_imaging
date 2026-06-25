# 完整FEM无监督训练启动指南

## 🎯 训练目标
修复雅可比线性近似导致的模型失效问题，使用完整FEM物理约束进行无监督训练。

## 📋 执行步骤

### 步骤1：启动训练
```bash
cd /home/ubuntu/EIT/eit_root_imaging
bash start_full_fem_training.sh
```

### 步骤2：监控训练状态
```bash
# 查看训练进程
ps aux | grep train_conv_spatial

# 实时查看日志
tail -f train_full_fem.log

# 检查GPU使用
nvidia-smi
```

### 步骤3：每10个epoch评估一次
```bash
# epoch 10评估
CUDA_VISIBLE_DEVICES="" python evaluate_current_run.py \
  --checkpoint checkpoints/latest/unsup_epoch10.pt \
  --data data/generated/mixed_dataset.h5 \
  --output results/eval_full_fem_epoch10 \
  --split val

# epoch 20评估
CUDA_VISIBLE_DEVICES="" python evaluate_current_run.py \
  --checkpoint checkpoints/latest/unsup_epoch20.pt \
  --data data/generated/mixed_dataset.h5 \
  --output results/eval_full_fem_epoch20 \
  --split val
```

## ⚙️ 训练参数说明
- `--mcl_mode full_fem`: 完整FEM物理约束（修复雅可比近似问题）
- `--batch_size 16`: 平衡GPU内存和训练速度
- `--grad_accum_steps 2`: 梯度累积，等效batch_size=32
- `--lr 5e-5`: 较小学习率，避免破坏预训练权重
- `--epochs_unsup 50`: 无监督训练轮数
- `--resume`: 从监督训练的best.pt恢复

## 📊 预期效果
- **训练前**: RE=0.193, CC=0.955
- **雅可比模式**: RE=0.648, CC=-0.01 (已失效)
- **FEM模式预期**: RE≤0.15, CC≥0.97

## ⏱️ 时间估算
- 单个epoch: ~15分钟 (full_fem模式较慢)
- 总训练时间: ~12.5小时 (50 epochs)
- 建议使用tmux保持会话

## 🔍 成功指标
- 训练损失稳步下降
- 验证RE持续改善
- 验证CC保持在0.95以上
- GPU利用率90%+

## ⚠️ 注意事项
1. 完整FEM模式较慢，每个batch约24ms
2. GPU内存使用约16GB
3. 检查点每10个epoch自动保存
4. 如遇OOM，减小batch_size至8

## 🐛 故障排查
如果训练失败：
1. 检查日志: `tail -100 train_full_fem.log`
2. 检查GPU内存: `nvidia-smi`
3. 检查数据文件: `ls -lh data/generated/mixed_dataset.h5`
4. 检查checkpoint: `ls -lh checkpoints/20260622_015538_v2_both_hd256/best.pt`
