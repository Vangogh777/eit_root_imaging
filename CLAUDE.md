# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

EIT (Electrical Impedance Tomography) **高精度、高泛化性通用成像算法** — 基于 pyEIT + PyTorch 的桶式 (bucket-type) 2D EIT 系统。目标是实现超越传统方法 (GN/GREIT) 的通用重建精度，适用于任意电导率分布（不限于根系）。当前以植物根系为主要测试场景，在单圈 16 电极圆柱容器中验证。

**Hardware context**: Cylindrical bucket (~20cm diameter), single ring of 16 electrodes, 2D cross-sectional imaging with multi-frequency measurements (6 frequencies: 1kHz-500kHz). **Generalization focus**: algorithm design targets arbitrary conductivity distributions (inclusions, anomalies, multi-phase materials), not plant-root-specific features.

**Config synchronization**: `mesh_config.yaml` (`mesh_resolution`) and `train_config.yaml` (`n_elems`) must match. Current config: `mesh_resolution: 0.004` → `n_elems: 4424`. Finer `mesh_resolution: 0.0025` → ~11466 elements.

## Commands

### Setup
```bash
cd eit_root_imaging
pip install -r requirements.txt
```

### Testing
```bash
python test_minimal.py                  # Quick pipeline validation (data → model → forward pass)
python test_traditional_methods.py      # Test traditional methods (GN, GREIT)
python test_residual_minimal.py         # Test residual EIT pipeline (Route B)
```

### Training

**ConvSpatialEIT (primary model, two-stage):**
```bash
python train_conv_spatial.py                                    # Default: 50 sup + 200 unsup epochs
python train_conv_spatial.py --epochs_sup 50 --epochs_unsup 200
python train_conv_spatial.py --mcl_mode full_fem                # Full FEM physics (fixes Jacobian approximation)
python train_conv_spatial.py --resume checkpoints/<run_id>/best.pt
python train_conv_spatial.py --batch_size 8 --grad_accum_steps 4  # Gradient accumulation for large models
```

**DiffEIT v3 (diffusion-based, experimental):**
```bash
python train_diff_eit.py                        # Full 2-phase: unconditional (100ep) + conditional (150ep)
python train_diff_eit.py --phase unconditional  # Unconditional pretraining only
python train_diff_eit.py --phase conditional    # Conditional finetuning only
python train_diff_eit.py --warm_start            # Enable warm-start residual diffusion
python train_diff_eit.py --resume <ckpt>         # Resume training
```

**Residual EIT (Route B — traditional inversion → neural correction):**
```bash
python train_residual_eit.py                    # Train residual correction model
```

**Convenience launchers:**
```bash
bash start_full_fem_training.sh                 # Full FEM + gradient accumulation (recommended)
bash run_training.sh                            # Two-stage with hidden_dim=512
bash run_train_v3.sh                            # ConvSpatialEIT v3 with model-Jacobian + GNN edge features
bash run_all.sh                                 # Two sequential ConvSpatialEIT runs
bash run_resume.sh                              # Resume from checkpoint
```

**Legacy models:**
```bash
python train.py --generate                      # Unsupervised SFSBLC
python train_server.py --model physics          # Physics-informed model
python train_server.py --model improved_gnn     # GNN with Jacobian prior
python train_m1.py --quick                      # M1 Mac quick test
```

**Training with email notification:**
```bash
# Monitor a running PID
python notify_train.py --pid 12345 --log train.log

# Launch training + notify on completion
python notify_train.py --log train_diff.log -- python train_diff_eit.py --phase conditional --epochs_cond 20
```

### Evaluation

```bash
# ConvSpatialEIT checkpoints
python evaluate_conv_spatial.py --checkpoint checkpoints/<run_id>/best.pt --data data/generated/mixed_dataset.h5
python evaluate_conv_spatial_v3.py --checkpoint checkpoints/<run_id>/best.pt   # Enhanced (more metrics, visualizations)
bash evaluate_checkpoint.sh checkpoints/<run_id>/                               # Batch evaluate all checkpoints

# Evaluate any run (reads meta.json, auto-builds correct model)
python evaluate_current_run.py --checkpoint checkpoints/<run_id>/best.pt --data data/generated/mixed_dataset.h5

# DiffEIT evaluation
python evaluate_conv_spatial_v3.py --checkpoint checkpoints/<run_id>/best.pt --model diff_eit

# Legacy SFSBLC
python evaluation/evaluate.py --checkpoint checkpoints/...pt --split test
```

### Visualization
```bash
python visualize_results.py                    # Default model
python visualize_results.py --model xxx.pt     # Specific checkpoint
```

### Inference
```python
from inference.inference import EITInference
engine = EITInference("checkpoints/model_final.pt")
sigma = engine(voltages)  # (n_freq, n_meas) → (n_elems,)
result = engine.predict_with_details(voltages)  # With intermediate outputs
```

### ONNX Export
```bash
python inference/onnx_export.py --checkpoint checkpoints/model_final.pt --output model.onnx
```

### Results Server
```bash
python serve_results.py                        # :8080 (live-updating results dashboard)
sudo python serve_results.py --port 80         # Production via systemd eit-server.service
```

## Architecture

### Model Families Overview

| Model | File | Lines | Description |
|-------|------|-------|-------------|
| **ConvSpatialEIT** | `models/conv_spatial_eit.py` | 634 | Two-stage GNN (supervised pretrain + unsupervised physics finetune). **Primary model.** |
| **DiffEIT v3** | `models/diff_eit.py` | 408 | Diffusion-based reconstruction with MeshUNet denoiser. **Experimental.** |
| **ResidualEIT** | `models/residual_eit.py` | 285 | Route B: traditional inversion → GNN residual correction |
| **SF-SBLC** | `models/sf_sblc.py` | — | Original unsupervised model family (legacy) |

### Primary Model: ConvSpatialEIT (`models/conv_spatial_eit.py`)

Two-stage training (supervised pretraining → unsupervised physics finetuning).

**Data flow**: `Voltages (B, 6, 208)` → `FrequencyCrossAttention` → `Conv2D Encoder` (13×16 grid) → `Grid Sampling + Position Encoding` → `GNN` → `Sigma (B, n_elems)`

Key components:
- `FrequencyCrossAttention` — multi-frequency fusion (replaces 1×1 Conv)
- `SEModule` — channel attention with squeeze-excitation
- GNN with positional encoding (radius + Fourier coordinate features)
- Optional `VoltageMasking` data augmentation (20% random mask)
- Optional GATv2 edge features + edge feature modulation (`--edge_ratio`)
- Optional model-Jacobian (`--use_model_jacobian`): predicts voltage correction alongside σ

Physics constraint modes (`--mcl_mode`):
- `full_fem` — Complete pyEIT forward solver (accurate; fixes Jacobian approximation issues for large σ deviations). Cached every N steps via `fem_interval: 5`
- `jacobian` — Fast linear approximation `V_pred ≈ J·(σ_pred - σ_ref)` (legacy)
- `hybrid` — Mix of both

**Historical best RE: 0.103** (hidden_dim=512, 2026-06-17).

### DiffEIT v3 — Diffusion Model (`models/diff_eit.py` + `models/mesh_unet.py` + `models/diffusion_utils.py` + `models/mesh_pooling.py`)

Diffusion-based reconstruction using DDPM with cosine schedule. Two-phase training:
1. **Phase 1 (unconditional)**: Learn σ prior distribution — only needs σ samples, not paired voltages
2. **Phase 2 (conditional)**: x₀-prediction with voltage/sensitivity conditioning + optional warm-start

**Data flow**: `Noise ε` + `Voltage encoding` + `Sensitivity features (J^T·V, J_energy)` + `Timestep t` → `MeshUNet (FiLM-conditioned)` → `σ_pred (x₀-prediction)`

Key components:
- `VoltageEncoder` — multi-frequency voltage encoding with cross-attention pooling
- `MeshUNet` — hierarchical GNN encoder-decoder on multi-scale graph hierarchy (FPS downsampling + k-NN pooling). FiLM conditioning at every level for time/voltage injection. Bottleneck cross-attention to voltage latent
- `DiffusionProcess` — DDPM with cosine noise schedule (narrow-range friendly for EIT σ ∈ [0.005, 0.1]). Supports both ε-prediction and x₀-prediction
- `build_hierarchy` (`models/mesh_pooling.py`) — Farthest Point Sampling + k-NN graph coarsening (pure PyTorch, no Graclus dependency)
- `--warm_start` — linear least-squares initial guess, then diffuse only the residual (more stable)
- Sensitivity features: `J^T·V` (back-projected measurement) and `J_energy` (per-element Jacobian column energy) injected as per-node input channels

### Residual EIT — Route B (`models/residual_eit.py`)

Two-stage alternative: **traditional reconstruction → neural residual correction**.

```
Voltages → GN/GREIT inversion → σ_0 (coarse)
                                  ↓
σ_0 → ResidualComputer (J·(σ_0 - σ_ref), J^T·residual) → ResidualMeshGNN → Δσ
                                  ↓
                            σ_final = σ_0 + Δσ
```

Key components:
- `ResidualComputer` — Jacobian residual features `J·(σ_0 - σ_ref)` and back-projected residual `J^T·(V - V_pred)`
- `VoltageGlobalEncoder` — encodes raw voltages into a global latent vector
- `ResidualMeshGNN` — GNN on FEM mesh predicting conductivity correction Δσ
- Training: `train_residual_eit.py` → `training/residual_trainer.py` → `training/residual_loss.py`
- Config: `config/residual_eit_config.yaml`
- Data: `data/generate_shapes_dataset.py` (circles, squares, etc.)

### Legacy Model: SF-SBLC (`models/sf_sblc.py`)

Original model family. Data flow: `SharedEncoder → BLC (BaseLayerCorrection) → FrequencyFusionDecoder → ResNetBackbone`.

Variants: `SFSBLC` (full), `SimpleSFSBLC` (simplified), `PhysicsInformedEIT` (`models/universal_eit.py`), `ImprovedEITModelGNN` (`models/improved_gnn_model.py`)

### Supporting Model Infrastructure

| Component | File | Purpose |
|-----------|------|---------|
| `layers/attention_gate.py` | Attention Gate | Spatial attention for feature refinement |
| `layers/spectral_conv.py` | Spectral Conv | Graph spectral convolution operations |
| `traditional/reconstructor.py` | Traditional Reconstructor | GN/GREIT reconstruction wrapper |
| `models/mesh_pooling.py` | Mesh Pooling | FPS + k-NN graph hierarchy for U-Net |
| `models/voltage_encoder.py` | Voltage Encoder | Shared voltage encoding for Route B |

### Training Philosophy

**Unsupervised (physics-constrained)** — no ground truth σ supervision:
`L_total = λ_m·L_meas + λ_tv·L_tv + λ_freq·L_freq + λ_blc·L_blc + λ_smooth·L_smooth + λ_dev·L_dev`

| Loss | Purpose |
|------|---------|
| `L_meas` | Measurement consistency: `‖F(σ_pred) - V_measured‖²` (core physics) |
| `L_tv` | Total variation (edge-preserving regularization) |
| `L_freq` | Frequency cross-consistency |
| `L_blc` | BLC correction constraint |
| `L_smooth` | Spatial smoothness |
| `L_dev` | Sigma deviation penalty (keeps σ near Jacobian linearization point) |

Measurement consistency: **Jacobian linear approximation** (fast) or **full FEM** (accurate, via pyEIT; default mode). Full FEM caches via `fem_interval: 5` (recompute every N steps).

**Supervised** — simple MSE between predicted and ground truth σ (used for pretraining or baselines).

**Diffusion (DiffEIT)**: x₀-prediction with MSE loss `‖σ̂_0 - σ_0‖²`, cosine noise schedule, optional warm-start residual diffusion.

## Data Pipeline

1. **Root dataset** (`data/generate_dataset.py`) — Random root structures (taproot/fibrous/herringbone) → pyEIT forward → `data/generated/eit_dataset.h5`
2. **Mixed dataset** (`data/generate_mixed_dataset.py`) — Multiple root types in one HDF5 → `data/generated/mixed_dataset.h5`
3. **Shapes dataset** (`data/generate_shapes_dataset.py`) — Geometric phantoms for residual EIT training → `data/generated/shapes_dataset/shapes_dataset.h5`
4. **Jacobian precomputation** (`data/precompute_jacobian.py`) → `data/generated/jacobian.npy`
5. **Residual features** (`data/precompute_residual_features.py`) — Precompute traditional reconstruction features for Route B
6. **PyEIDORS bridge** (`data/pyeidors_generate_data.py`) — MATLAB-bridge data generation
7. **Dataset classes** (`data/datasets/eit_dataset.py`):
   - `MemoryEITDataset` — loads entire HDF5 into RAM (fast, for datasets < 2GB)
   - `EITDataset` — memory-mapped HDF5 access (for larger datasets)
8. **EIT forward solver** (`data/eit_forward.py`) — pyEIT forward wrapper; `root_simulator.py` for phantom generation

## Configuration

- `config/mesh_config.yaml` — Mesh geometry (radius=0.10m, mesh_resolution=0.004 → ~6000-7000 elements, 4424 used in training), 16 electrodes, 6 frequencies, σ_soil=0.01 S/m, σ_root=0.05 S/m
- `config/train_config.yaml` — Model hyperparams (n_elems=4424, hidden_dim=512, n_res_blocks=8), loss weights, MCL mode, FEM interval, train/val/test sample counts
- `config/mesh_fine_config.yaml` — Higher resolution mesh (mesh_resolution=0.0025 → ~11466 elements)
- `config/residual_eit_config.yaml` — Route B model and training hyperparameters
- `config/pyeidors_train_config.yaml` — PyEIDORS-specific config

**Config synchronization rule**: `mesh_config.yaml` `mesh_resolution` must match `train_config.yaml` `n_elems`. Current: `0.004` → 4424 elements. Mismatch causes dimension errors.

## Key Patterns

- All paths in configs are relative to `eit_root_imaging/` directory
- `train_conv_spatial.py` auto-generates data only on first run
- **Gradient accumulation**: `--batch_size 8 --grad_accum_steps 4` yields effective batch size 32 (needed for large models on limited GPU memory)
- **FEM caching**: Full FEM mode recomputes forward solve every `fem_interval` steps (default: 5), caching Jacobian intermediate results between updates
- **EMA**: Use `--ema_decay 0.999` for long unsupervised training runs to stabilize
- Checkpoints use per-run isolation: `checkpoints/<run_id>/best.pt`, `checkpoints/<run_id>/final.pt`, `checkpoints/<run_id>/unsup_epoch*.pt`
- Checkpoint format varies — use `extract_model_state()` helper: newer scripts use `{'model': state_dict}`, older use `{'model_state_dict': state_dict}`
- `evaluate_conv_spatial.py` / `evaluate_conv_spatial_v3.py` for ConvSpatialEIT; `evaluation/evaluate.py` for SFSBLC
- Training records stored in `training_records/<run_id>/` with `index.json` as catalog; `training_records/current` symlinks to the latest run (used by serve_results.py for live dashboards)
- **Current training status** (2026-06-25): Multiple v2_both_hd256 and residual_eit runs; DiffEIT v3 training initiated
- `serve_results.py` is the live results dashboard — scans `results/`, `docs/`, and `training_records/`; deployed via systemd `eit-server.service` on port 80
- `notify_train.py` sends email on training completion (SMTP via env vars: `SMTP_HOST`, `SMTP_PORT`, `SMTP_USER`, `SMTP_PASS`, `NOTIFY_TO`)
- Model intermediate outputs: `base_map`, `freq_weights`, `blc_gates`
- pyEIT 1.2.4: frequency parameter is ignored in forward solver (multi-frequency returns identical copies) — multi-frequency training is for architecture robustness, not true spectral data
- Training scripts support `--wandb` flag for Weights & Biases logging; TensorBoard by default (`logs/` directory)
- GPU server background training: use `tmux new -s eit` or `nohup python train_conv_spatial.py ... > train.log 2>&1 &`
- `TRAINING_GUIDE.md` documents the full-FEM unsupervised training procedure; `docs/` contains design docs and analysis reports
