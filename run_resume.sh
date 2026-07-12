#!/bin/bash
cd /home/ubuntu/EIT/eit_root_imaging
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
exec python -u train_conv_spatial.py \
  --resume checkpoints/20260624_223848_v2_both_hd256/best.pt \
  --hidden_dim 256 \
  --mode both \
  --epochs_sup 50 \
  --epochs_unsup 200 \
  --use_model_jacobian \
  --batch_size 16 \
  --grad_accum_steps 2 \
  --mcl_mode jacobian \
  >> train_conv_spatial_resume.log 2>&1
