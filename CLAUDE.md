# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

EIT (Electrical Impedance Tomography) plant root unsupervised imaging system using pyEIT + PyTorch. This is a **桶式 (bucket-type) 2D EIT system** for imaging plant roots in cylindrical containers with a single ring of 16 electrodes.

**Hardware context**: Cylindrical bucket (~20cm diameter), single ring of 16 electrodes, 2D cross-sectional imaging with multi-frequency measurements (6 frequencies: 1kHz-500kHz).

## Commands

### Setup
```bash
cd eit_root_imaging
pip install -r requirements.txt
```

### Training
```bash
# First run (auto-generates data, then trains)
python train.py

# Subsequent runs (reuses existing data)
python train.py --epochs 300 --batch_size 64

# Force regenerate data
python train.py --generate

# Resume from checkpoint
python train.py --resume checkpoints/eit_20260610/checkpoint_epoch_50.pt
```

### Evaluation
```bash
python evaluation/evaluate.py --checkpoint checkpoints/...pt --split test
```

### Inference
```python
from inference.inference import EITInference
engine = EITInference("checkpoints/model_final.pt")
sigma = engine(voltages)  # voltages: (n_freq, n_meas) → sigma: (n_elems,)
```

## Architecture

### Core Model: SF-SBLC (Spatial-Frequency Shared and Base Layer Correction)

The main model (`models/sf_sblc.py`) combines four components:
1. **SharedEncoder** - Multi-frequency shared encoding
2. **BaseLayerCorrection (BLC)** - Suppresses system artifacts
3. **FrequencyFusionDecoder** - Fuses multi-frequency features
4. **ResNetBackbone** - Deep residual reconstruction

**Data flow**:
```
Voltages (B, n_freq, n_meas) → SharedEncoder → BLC → FusionDecoder → ResNetBackbone → Sigma (B, n_elems)
```

**Variants**:
- `SFSBLC` - Full model with all components
- `SFSBLC_Light` - Lightweight version for quick prototyping (fewer parameters)

### Unsupervised Training Philosophy

This system uses **physics-constrained unsupervised learning** - no ground truth σ is used as supervision during training:

**Loss function**: `L_total = λ_m * L_meas + λ_tv * L_tv + λ_freq * L_freq + λ_blc * L_blc + λ_smooth * L_smooth`

| Loss | Purpose |
|------|---------|
| `L_meas` | Measurement consistency: `‖F(σ_pred) - V_measured‖²` (core physics constraint) |
| `L_tv` | Total variation regularization (suppresses artifacts, preserves edges) |
| `L_freq` | Frequency cross-consistency (multi-frequency structure agreement) |
| `L_blc` | BLC correction constraint (prevents over-correction) |
| `L_smooth` | Spatial smoothness (physical plausibility) |

The measurement consistency loss uses either:
- **Jacobian linear approximation** (fast, for training): `V_pred ≈ J · (σ_pred - σ_ref)`
- **Full pyEIT forward solver** (accurate, for validation)

### Key Modules

| Module | Purpose |
|--------|---------|
| `data/eit_forward.py` | pyEIT FEM forward solver, multi-frequency simulation |
| `data/root_simulator.py` | Random root structure generation (taproot/fibrous/herringbone) |
| `data/generate_dataset.py` | Generates HDF5 datasets with simulated measurements |
| `training/unsupervised_loop.py` | Main training loop with physics-constrained losses |
| `training/loss.py` | All loss functions (measurement consistency, TV, frequency cross-consistency, etc.) |
| `inference/inference.py` | Production inference engine |
| `inference/onnx_export.py` | ONNX export for deployment |

### Configuration

- `config/mesh_config.yaml`: Mesh (radius=10cm, resolution=5mm), electrodes (16), stimulation frequencies (6), conductivity values (soil=0.01 S/m, root=0.05 S/m)
- `config/train_config.yaml`: Model hyperparameters, training settings, loss weights

### Data Pipeline

1. **Dataset generation** (`data/generate_dataset.py`):
   - Creates random root structures via `RootSystemGenerator`
   - Runs pyEIT forward simulation to get boundary voltages
   - Stores in HDF5 format: `data/generated/eit_dataset.h5`

2. **Jacobian precomputation** (`data/precompute_jacobian.py`):
   - Precomputes sensitivity matrix for fast training
   - Output: `data/generated/jacobian.npy`

3. **EITDataset** (`data/datasets/eit_dataset.py`):
   - PyTorch Dataset/DataLoader for HDF5 files
   - Handles train/val/test splits

### Mesh Structure

- 2D circular domain (bucket cross-section)
- ~1500 triangular elements, ~800 nodes
- Element centers used for spatial operations (TV regularization, visualization)

### Evaluation Metrics

- **RE**: Relative error `‖pred - target‖ / ‖target‖`
- **CC**: Correlation coefficient
- **SSIM**: Structural similarity (computed on mesh-interpolated images)

## Important Patterns

- All paths in configs are relative to `eit_root_imaging/` directory
- The `train.py` entry point handles data generation automatically - data is only generated once
- Checkpoints saved to `checkpoints/eit_{date}/`
- TensorBoard logging enabled by default
- Model outputs include interpretable intermediate results: `base_map`, `freq_weights`, `blc_gates`
