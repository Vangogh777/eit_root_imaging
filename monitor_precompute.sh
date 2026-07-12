#!/bin/bash
# 监控预计算进度

echo "=== 预计算进度监控 ==="
echo "开始时间: $(date)"
echo ""

# 等待预计算完成
while ps aux | grep -q "[p]recompute_residual_features"; do
    clear
    echo "=== 预计算进度 ==="
    echo "时间: $(date)"
    echo ""

    # 显示最新日志
    tail -5 precompute_mixed.log 2>/dev/null || tail -5 /tmp/claude-1000/-home-ubuntu-EIT-eit-root-imaging/f37a4262-a6b9-458e-bf5a-078a3bc0818a/tasks/blkxlf84o.output 2>/dev/null

    echo ""
    echo "按 Ctrl+C 取消监控"
    sleep 5
done

echo ""
echo "=== 预计算完成！==="
echo "完成时间: $(date)"
echo ""

# 验证预计算结果
echo "=== 验证预计算结果 ==="
python3 << 'EOF'
import h5py

h5_path = "data/generated/mixed_dataset.h5"
with h5py.File(h5_path, 'r') as f:
    print(f"\n数据集: {h5_path}")
    for split in ['train', 'val', 'test']:
        if split in f:
            has_features = all(name in f[split] for name in ('sigma_0', 'physics_g', 'voltage_residual'))
            n_samples = f[split]['voltages'].shape[0]
            status = "✅ 已有残差特征" if has_features else "❌ 缺少残差特征"
            print(f"  {split}: {n_samples} 样本, {status}")
EOF

echo ""
echo "=== 准备启动训练 ==="
echo "配置文件: config/residual_eit_config.yaml"
echo "数据集: mixed_dataset.h5 (20000训练样本)"
echo ""
echo "启动命令: python train_residual_eit.py"
