# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

EIT (Electrical Impedance Tomography) plant root unsupervised imaging system using pyEIT + PyTorch. This is a **桶式 (bucket-type) 2D EIT system** for imaging plant roots in cylindrical containers with a single ring of 16 electrodes.

**Hardware context**: Cylindrical bucket (~20cm diameter), single ring of 16 electrodes, 2D cross-sectional imaging with multi-frequency measurements (6 frequencies: 1kHz-500kHz).

**Config synchronization**: `mesh_config.yaml` (`mesh_resolution`) and `train_config.yaml` (`n_elems`) must match. Current config: `mesh_resolution: 0.004` → `n_elems: 4424`. Finer `mesh_resolution: 0.0025` → ~11466 elements.

## Commands

### Setup
```bash
cd eit_root_imaging
pip install -r requirements.txt
```

### Testing
```bash
# Quick pipeline validation (data generation → model → forward pass)
python test_minimal.py

# Test traditional reconstruction methods (GN, GREIT)
python test_traditional_methods.py

# Test residual EIT pipeline (Route B)
python test_residual_minimal.py
```

### Training

**ConvSpatialEIT (primary model, two-stage):**
```bash
python train_conv_spatial.py                              # Default: 50 sup + 200 unsup epochs
python train_conv_spatial.py --epochs_sup 50 --epochs_unsup 200
python train_conv_spatial.py --mcl_mode full_fem          # Full FEM physics (fixes Jacobian approximation issue)
python train_conv_spatial.py --resume checkpoints/<run_id>/best.pt  # Resume unsupervised phase
```

**Full-FEM unsupervised training** (recommended for best results — see `TRAINING_GUIDE.md`):
```bash
bash start_full_fem_training.sh                           # Full FEM + gradient accumulation
```

**Residual EIT (Route B — traditional inversion → neural correction):**
```bash
python train_residual_eit.py                              # Train residual correction model
```

**Legacy models:**
```bash
python train.py                         # Unsupervised SFSBLC (auto-generates data on first run)
python train.py --generate              # Force regenerate data
python train.py --resume checkpoints/...pt

python train_server.py --model physics         # Physics-informed model
python train_server.py --model improved_gnn    # GNN with Jacobian prior
python train_server.py --model simple          # Simple MLP

python train_m1.py --quick                     # M1 Mac quick test
```

### Evaluation

```bash
# Primary: evaluate ConvSpatialEIT checkpoints
python evaluate_conv_spatial.py --checkpoint checkpoints/<run_id>/best.pt --data data/generated/mixed_dataset.h5

# Evaluate current run (reads meta.json, auto-builds correct model)
python evaluate_current_run.py --checkpoint checkpoints/<run_id>/best.pt --data data/generated/mixed_dataset.h5

# V3 enhanced evaluation (more metrics, visualizations)
python evaluate_conv_spatial_v3.py --checkpoint checkpoints/<run_id>/best.pt

# Batch evaluate all checkpoints in a run
bash evaluate_checkpoint.sh checkpoints/<run_id>/

# Legacy SFSBLC evaluation
python evaluation/evaluate.py --checkpoint checkpoints/...pt --split test
```

### Visualization
```bash
python visualize_results.py                    # Use default model
python visualize_results.py --model xxx.pt    # Specify model checkpoint
```

### Inference
```python
from inference.inference import EITInference
engine = EITInference("checkpoints/model_final.pt")
sigma = engine(voltages)  # voltages: (n_freq, n_meas) → sigma: (n_elems,)

# With intermediate outputs
result = engine.predict_with_details(voltages)
```

### ONNX Export
```bash
python inference/onnx_export.py --checkpoint checkpoints/model_final.pt --output model.onnx
```

### Results Server
```bash
python serve_results.py             # HTTP server on :8080 (live-updating results page)
sudo python serve_results.py --port 80  # Production via systemd eit-server.service
```

## Architecture

### Primary Model: ConvSpatialEIT (`models/conv_spatial_eit.py`)

Current main model (634 lines). Two-stage training:
1. **Stage 1 - Supervised pretraining**: MSE on paired data (voltage → σ), quick convergence
2. **Stage 2 - Unsupervised finetuning**: Physics constraints (measurement consistency)

**Data flow**: `Voltages (B, 6, 208)` → `FrequencyCrossAttention` → `Conv2D Encoder` (13×16 grid) → `Grid Sampling + Position Encoding` → `GNN` → `Sigma (B, n_elems)`

Key architectural components:
- `FrequencyCrossAttention` — replaces 1×1 Conv for multi-frequency fusion
- `SEModule` — channel attention with squeeze-excitation
- GNN with positional encoding (radius + Fourier coordinate features)
- Optional `VoltageMasking` data augmentation (15% random mask)

Physics constraint modes (`--mcl_mode`):
- `jacobian` — Fast linear approximation `V_pred ≈ J·(σ_pred - σ_ref)` (may fail for large σ deviations)
- `full_fem` — Complete pyEIT forward solver (accurate, slower; fixes Jacobian approximation issues)
- `hybrid` — Mix of both

**Historical best RE: 0.103** (hidden_dim=512, 2026-06-17).

### Residual EIT — Route B (`models/residual_eit.py`)

Two-stage alternative approach: **traditional reconstruction → neural residual correction**.

```
Voltages → GN/GREIT inversion → σ_0 (coarse)
                                  ↓
σ_0 → ResidualComputer (J·(σ_0 - σ_ref), J^T·residual) → ResidualMeshGNN → Δσ
                                  ↓
                            σ_final = σ_0 + Δσ
```

Key components:
- `ResidualComputer` — computes Jacobian residual features `J·(σ_0 - σ_ref)` and back-projected residual `J^T·(V - V_pred)`
- `VoltageGlobalEncoder` — encodes raw voltages into a global latent vector
- `ResidualMeshGNN` — GNN operating on FEM mesh to predict conductivity correction Δσ
- Training: `train_residual_eit.py` → `training/residual_trainer.py` → `training/residual_loss.py`
- Config: `config/residual_eit_config.yaml`
- Data: `data/generate_shapes_dataset.py` generates shapes phantoms (circles, squares, etc.) for training

### Legacy Model: SF-SBLC (`models/sf_sblc.py`)

Original model family. Data flow: `SharedEncoder → BLC (BaseLayerCorrection) → FrequencyFusionDecoder → ResNetBackbone`.

Variants:
- `SFSBLC` — full model (used by `train.py`)
- `SimpleSFSBLC` — simplified for supervised training (used by `train_m1.py`, `train_server.py`)
- `PhysicsInformedEIT` — physics-informed with Jacobian prior (`models/universal_eit.py`)
- `ImprovedEITModelGNN` — GNN with mesh structure + Jacobian features (`models/improved_gnn_model.py`)

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
| `L_dev` | Sigma deviation penalty |

Measurement consistency source: **Jacobian linear approximation** (fast, for training) or **full FEM** (accurate). The full FEM mode was added to fix cases where the Jacobian linearization fails for large conductivity deviations.

**Supervised** — simple MSE between predicted and ground truth σ (used for pretraining or baselines).

## Data Pipeline

1. **Root dataset** (`data/generate_dataset.py`) — Random root structures (taproot/fibrous/herringbone) → pyEIT forward → `data/generated/eit_dataset.h5`
2. **Mixed dataset** (`data/generate_mixed_dataset.py`) — Multiple root types in one HDF5 → `data/generated/mixed_dataset.h5`
3. **Shapes dataset** (`data/generate_shapes_dataset.py`) — Geometric phantoms for residual EIT training → `data/generated/shapes_dataset/shapes_dataset.h5`
4. **Jacobian precomputation** (`data/precompute_jacobian.py`) → `data/generated/jacobian.npy`
5. **Residual features** (`data/precompute_residual_features.py`) — Precompute traditional reconstruction features for Route B
6. **Dataset classes** (`data/datasets/eit_dataset.py`):
   - `MemoryEITDataset` — loads entire HDF5 into RAM (fast, for datasets < 2GB)
   - `EITDataset` — memory-mapped HDF5 access (for larger datasets)

## Configuration

- `config/mesh_config.yaml` — Mesh (radius=0.10m, mesh_resolution=0.004 → ~6000-7000 elements, 4424 used in training), 16 electrodes, 6 frequencies, σ_soil=0.01 S/m, σ_root=0.05 S/m
- `config/mesh_fine_config.yaml` — Higher resolution mesh (mesh_resolution=0.0025 → ~11466 elements)
- `config/residual_eit_config.yaml` — Route B model and training hyperparameters
- Various unused configs in `config/` may exist for different experiments

## Key Patterns

- All paths in configs are relative to `eit_root_imaging/` directory
- `train_conv_spatial.py` auto-generates data only on first run
- Checkpoints use per-run isolation: `checkpoints/<run_id>/best.pt`, `checkpoints/<run_id>/final.pt`, `checkpoints/<run_id>/unsup_epoch*.pt`
- Checkpoint format varies — use `extract_model_state()` helper: newer scripts use `{'model': state_dict}`, older use `{'model_state_dict': state_dict}`
- `evaluate_conv_spatial.py` for ConvSpatialEIT; `evaluation/evaluate.py` for SFSBLC
- Training records stored in `training_records/<run_id>/` with `index.json` as catalog; `training_records/current` symlinks to the latest run
- `serve_results.py` is the live results dashboard — scans `results/`, `docs/`, and `training_records/`; deployed via `eit-server.service` systemd unit
- Model intermediate outputs: `base_map`, `freq_weights`, `blc_gates`
- pyEIT 1.2.4: frequency parameter is ignored in forward solver (multi-frequency returns identical copies)
- Training scripts support `--wandb` flag for Weights & Biases logging; TensorBoard by default (`logs/` directory)
- GPU server background training: use `nohup python train_conv_spatial.py ... > train.log 2>&1 &` or `tmux`
- `TRAINING_GUIDE.md` documents the full-FEM unsupervised training procedure
