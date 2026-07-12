# AGENTS.md

This file provides guidance to Reasonix/Codex when working with code in this repository.

## Project Overview

EIT (Electrical Impedance Tomography) high-precision plant root imaging using pyEIT + PyTorch. **桶式 (bucket-type) 2D EIT system** — single ring of 16 electrodes, 2D cross-section, 6 frequencies (1kHz-500kHz).

**Stack**: Python, PyTorch, pyEIT (FEM forward solver), HDF5, ONNX, wandb, TensorBoard.

## Commands

### Setup
```bash
cd eit_root_imaging
pip install -r requirements.txt
```

### Quick Validation
```bash
python test_minimal.py              # Data gen → model → forward pass
```

### Training (Entry Points)
```bash
# ConvSpatialEIT (primary, two-stage: sup + unsup)
python train_conv_spatial.py        # Default: 50 sup + 200 unsup epochs
python train_conv_spatial.py --mcl_mode full_fem --batch_size 8 --grad_accum_steps 4

# DiffEIT v5 (single-phase conditional diffusion)
python train_diff_eit.py            # Default: T=50, hidden_dim=384, 150 epochs
python train_diff_eit.py --T 100 --hidden_dim 512 --epochs 200

# Residual EIT (Route B: traditional → GNN correction)
python train_residual_eit.py

# Legacy SFSBLC
python train.py                     # Unsupervised (auto-generates data on first run)
python train_server.py --model physics  # Physics-informed

# Convenience
bash start_full_fem_training.sh     # Full FEM + gradient accumulation
bash run_training.sh                # Two-stage with hidden_dim=512
bash run_per_shape_training.sh      # Sequential training across 5 shapes (11466 mesh)
```

### Evaluation
```bash
# Auto-detect model type from checkpoint (recommended)
python evaluate_current_run.py --checkpoint checkpoints/<run_id>/best.pt --data data/generated/mixed_dataset.h5

# Model-specific
python evaluate_conv_spatial_v3.py --checkpoint checkpoints/<run_id>/best.pt
python evaluate_diff_eit.py --checkpoint checkpoints/<run_id>/best.pt
python evaluation/evaluate.py --checkpoint <ckpt> --split test  # Legacy SFSBLC

# Batch
bash evaluate_checkpoint.sh checkpoints/<run_id>/
```

### Visualization & Server
```bash
python visualize_results.py
python serve_results.py             # :8080 (live dashboard)
```

### Inference
```python
from inference.inference import EITInference
engine = EITInference("checkpoints/model_final.pt")
sigma = engine(voltages)  # (n_freq, n_meas) → (n_elems,)
```

## Architecture

### Core Model Families

| Model | File | Description |
|-------|------|-------------|
| **ConvSpatialEIT** | `models/conv_spatial_eit.py` | Two-stage GNN (supervised + unsupervised). **Primary model.** v3 adds PAMP: Jacobian互灵敏度调制GNN消息。Best RE: 0.073 (ellipse, 11466). |
| **DiffEIT v5** | `models/diff_eit.py` | Diffusion with MeshUNet + RankGauss + CFG. **Deprecated** — RE 0.95, 13x worse than ConvSpatialEIT. |
| **ResidualEIT** | `models/residual_eit.py` | Route B: GN/GREIT → GNN residual correction |
| **SF-SBLC** | `models/sf_sblc.py` | Original unsupervised model (legacy) |

### Training Modes

- **Unsupervised (ConvSpatialEIT)**: Physics-constrained — `L_total = λ_m·L_meas + λ_tv·L_tv + λ_freq·L_freq + λ_blc·L_blc + λ_smooth·L_smooth + λ_dev·L_dev`. Uses Jacobian linear approximation or full FEM for measurement consistency.
- **Supervised (pretraining)**: Simple MSE between predicted and ground truth σ.
- **Diffusion (DiffEIT v5)**: x₀-prediction with MSE in RankGauss N(0,1) space + physics consistency loss. DDIM sampling with CFG.

### Data Pipeline
1. `data/generate_dataset.py` → `data/generated/eit_dataset.h5`
2. `data/generate_mixed_dataset.py` → `data/generated/mixed_dataset.h5`
3. `data/precompute_jacobian.py` → `data/generated/jacobian.npy`
4. `data/datasets/eit_dataset.py` — `MemoryEITDataset` (RAM) / `EITDataset` (memory-mapped)
5. `data/eit_forward.py` — pyEIT forward wrapper; `root_simulator.py` for phantom generation

### Configuration
- `config/mesh_config.yaml` + `config/train_config.yaml` — Coarse mesh: 4424 elements (mesh_resolution=0.004)
- `config/mesh_11466_config.yaml` + `config/train_config_11466.yaml` — Fine mesh: 11466 elements (mesh_resolution=0.0025)
- `config/residual_eit_config.yaml` — Route B config
- **Rule**: `mesh_resolution` and `n_elems` must match across mesh + train configs

### Evaluation Metrics
- **RE**: Relative error `‖pred - target‖ / ‖target‖`
- **CC**: Correlation coefficient
- **SSIM**: Structural similarity (mesh-interpolated images)

### Training Records & Results Server
- **TrainingRecorder** (`training/recorder.py`): Logs each run's config, per-epoch metrics, events to `training_records/{run_id}/`
- **serve_results.py**: HTTP dashboard scanning `results/`, `docs/`, and `training_records/`
- **eit-server.service**: systemd unit for production deployment on port 80

## Key Patterns

- All paths in configs are relative to `eit_root_imaging/`
- `train_conv_spatial.py` auto-generates data only on first run
- Checkpoints: `checkpoints/<run_id>/best.pt`, `final.pt`, `unsup_epoch*.pt`
- Checkpoint format varies — `{'model': state_dict}` (newer) vs `{'model_state_dict': state_dict}` (older)
- DiffEIT checkpoints include persistent RankGauss buffers — load with appropriate handling
- Training records: `training_records/<run_id>/` with `index.json` catalog; `current` symlinks to latest
- Full FEM mode caches forward solve via `fem_interval: 5`
- EMA (`--ema_decay 0.999`) recommended for long unsupervised runs
- Use `tmux` or `nohup` for background GPU training
- `notify_train.py` — email notification on training completion
- pyEIT 1.2.4: frequency parameter is ignored in forward solver
- torch.compile used in DiffEIT for MeshUNet (~20-30% speedup)
