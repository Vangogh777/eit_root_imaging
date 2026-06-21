# AGENTS.md

This file provides guidance to Reasonix/Codex when working with code in this repository.

## Project Overview

EIT (Electrical Impedance Tomography) plant root unsupervised imaging system using pyEIT + PyTorch. **桶式 (bucket-type) 2D EIT system** with a single ring of 16 electrodes around a cylindrical container (~20cm diameter), multi-frequency measurements (6 frequencies: 1kHz-500kHz).

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
python train.py                     # Default: SFSBLC model (auto-generates data on first run)
python train.py --generate          # Force regenerate data
python train.py --resume <ckpt>     # Resume from checkpoint

python train_conv_spatial.py        # Two-stage: supervised pretrain + unsupervised finetune
python train_conv_spatial.py --epochs_sup 50 --epochs_unsup 200

python train_two_stage.py           # Traditional inversion → neural refinement
python train_two_stage.py --refine_type unet --wandb

python train_pyeidors.py            # Train using PyEIDORS-generated data (MATLAB bridge)

python train_m1.py                  # M1 Mac optimized (MPS acceleration)
python train_m1.py --quick          # Quick test: 100 samples, 10 epochs

python train_server.py              # GPU server training with multiple model choices
python train_server.py --n_train 20000 --model physics

python run_all.sh                   # Run two Conv-Spatial training rounds sequentially
```

### Evaluation & Visualization
```bash
python evaluation/evaluate.py --checkpoint <ckpt> --split test
python evaluation/validate.py --checkpoint <ckpt>
python evaluation/validate_conv_spatial.py --checkpoint <ckpt>
python evaluate_conv_spatial.py     # Conv-spatial specific evaluation
python evaluate_conv_spatial_full.py
python visualize_results.py         # Visualize reconstructions (uses default or --model)
```

### Results Server
```bash
python serve_results.py             # HTTP server on :8080 (live-updating results page)
python serve_results.py --port 80   # Production via systemd (eit-server.service)
```

### Inference
```python
from inference.inference import EITInference
engine = EITInference("checkpoints/model_final.pt")
sigma = engine(voltages)  # voltages: (n_freq, n_meas) → sigma: (n_elems,)
```

## Architecture

### Core Model Families

| Model | File | Description |
|-------|------|-------------|
| **SF-SBLC** (`SFSBLC`) | `models/sf_sblc.py` | Spatial-Frequency Shared + Base Layer Correction — the original model. SharedEncoder → BLC → FusionDecoder → ResNetBackbone |
| **ConvSpatialEIT** | `models/conv_spatial_eit.py` | Convolutional spatial model with two-stage (supervised+unsupervised) training |
| **TwoStageEITModel** | `models/two_stage_model.py` | Traditional inversion (GN/GREIT) → neural refinement (UNet/Graph) |
| **PhysicsInformedEIT** | `models/universal_eit.py` | Physics-informed with PDE constraints |
| **EITModelGNN** | `models/eit_gnn_model.py` | Graph Neural Network on FEM mesh |
| **ImprovedEITModelGNN** | `models/improved_gnn_model.py` | Enhanced GNN with attention |
| **SimpleSFSBLC** | `models/simple_model.py` | Lightweight for quick tests |
| **LinearEITModel / DeepEITModel** | `models/linear_model.py` | Linear/deep baselines |
| **PhysicsGNN** | `models/physics_gnn.py` | Physics-constrained GNN |

### Unsupervised Training Philosophy

Physics-constrained unsupervised learning — no ground truth σ used as supervision. Core loss:

`L_total = λ_m * L_meas + λ_tv * L_tv + λ_freq * L_freq + λ_blc * L_blc + λ_smooth * L_smooth + λ_dev * L_dev`

| Loss | Purpose |
|------|---------|
| `L_meas` | Measurement consistency: `‖F(σ_pred) - V_measured‖²` (core physics) |
| `L_tv` | Total variation (edge-preserving regularization) |
| `L_freq` | Frequency cross-consistency |
| `L_blc` | BLC correction constraint |
| `L_smooth` | Spatial smoothness |
| `L_dev` | Sigma deviation penalty (keeps σ near Jacobian linearization point) |

Measurement consistency uses either **Jacobian linear approximation** (fast) or **full FEM** (accurate, via pyEIT).

### Training Variants
- **train.py**: E2E unsupervised (original SFSBLC pipeline)
- **train_conv_spatial.py**: Two-stage — first supervised MSE on paired data, then unsupervised physics finetune
- **train_two_stage.py**: Two-stage — traditional inversion (GN/GREIT) as first pass, neural refinement second
- **train_pyeidors.py**: Uses data from MATLAB PyEIDORS pipeline

### Data Pipeline
1. `data/generate_dataset.py` — Random root structures → pyEIT forward → HDF5 (`data/generated/eit_dataset.h5`)
2. `data/generate_mixed_dataset.py` — Multiple root types (taproot/fibrous/herringbone) in one HDF5
3. `data/precompute_jacobian.py` — Sensitivity matrix (`data/generated/jacobian.npy`)
4. `data/generate_circle_dataset.py` / `generate_square_dataset.py` — Alternative domain shapes
5. `data/datasets/eit_dataset.py` — PyTorch Dataset/DataLoader (`MemoryEITDataset`, `EITDataset`)
6. `data/pyeidors_generate_data.py` / `data/pyeidors_data_generator.py` — MATLAB-bridge data generation

### Training Records & Results Server
- **TrainingRecorder** (`training/recorder.py`): Logs each run's config, per-epoch metrics, events to `training_records/{run_id}/`
- **serve_results.py**: HTTP server that scans `results/` and `training_records/` for live web display
- **eit-server.service**: systemd unit for production deployment on port 80

### Configuration
- `config/mesh_config.yaml`: Mesh geometry (radius=0.10m, mesh_resolution=0.0025, n_elems~11466 for h0=2.5mm), 16 electrodes, 6 frequencies
- `config/train_config.yaml`: Model hyperparams (n_elems=4424, hidden_dim=512, n_res_blocks=8), loss weights, data paths
- `config/pyeidors_train_config.yaml`: PyEIDORS-specific config

### Mesh Structure
- 2D circular domain (bucket cross-section)
- Configurable resolution: `mesh_resolution: 0.0025` → ~11466 elements; `h0: 0.004` (in train_config) → 4424 elements
- Element centers used for spatial operations (TV regularization, visualization)

### Evaluation Metrics
- **RE**: Relative error `‖pred - target‖ / ‖target‖`
- **CC**: Correlation coefficient
- **SSIM**: Structural similarity (mesh-interpolated images)

## Dependencies (core)
`torch`, `torchvision`, `numpy`, `scipy`, `pyEIT`, `matplotlib`, `meshio`, `tensorboard`, `tqdm`, `pyyaml`, `wandb`, `scikit-image`, `pandas`, `h5py`, `onnx`, `onnxruntime`

## Important Patterns

- All paths in configs are relative to `eit_root_imaging/` directory
- The `train.py` entry point auto-generates data only on first run (checks `data/generated/eit_dataset.h5`)
- Checkpoints saved to `checkpoints/` with naming like `conv_spatial_best.pt`, `two_stage_model.pt`
- Training records stored in `training_records/` with `index.json` as catalog
- Wandb logging available via `--wandb` flag; TensorBoard by default
- `serve_results.py` is the live results dashboard — deploy with `eit-server.service`
- Model outputs may include interpretable intermediates: `base_map`, `freq_weights`, `blc_gates`
- Mesh resolution differs between configs: `mesh_config.yaml` (h0=0.0025 → ~11466 cells) vs `train_config.yaml` (n_elems=4424, corresponding to h0=0.004)

## Notes
