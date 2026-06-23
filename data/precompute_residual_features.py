"""
Precompute residual-route features for an EIT HDF5 dataset.

Writes the following datasets into each split:
  - sigma_0: traditional coarse reconstruction
  - physics_g: normalized J^T r feature
  - voltage_residual: r = V_diff - J(sigma_0 - sigma_ref)
  - coarse_residual_norm: ||r|| / ||V_diff||
"""

import argparse
import os
import sys

import h5py
import numpy as np
from tqdm import tqdm

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from data.eit_forward import EITForwardSolver
from models.traditional import build_reconstructor


def load_or_compute_jacobian(path, solver):
    if path and os.path.exists(path):
        J = np.load(path).astype(np.float32)
    else:
        J = solver.get_jacobian().astype(np.float32)
        if path:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            np.save(path, J)
    if J.ndim == 3:
        J = J[0]
    return J


def recreate_dataset(group, name, data, compression="gzip"):
    if name in group:
        del group[name]
    group.create_dataset(name, data=data, compression=compression)


def split_has_features(group):
    return all(name in group for name in ("sigma_0", "physics_g", "voltage_residual"))


def process_split(group, split_name, solver, reconstructor, J, sigma_ref, force=False):
    if split_has_features(group) and not force:
        print(f"  [{split_name}] 已存在残差特征，跳过。使用 --force 可重算")
        return

    voltages = group["voltages"]
    n_samples = voltages.shape[0]
    n_elems = J.shape[1]
    n_meas = J.shape[0]

    sigma_0 = np.zeros((n_samples, n_elems), dtype=np.float32)
    physics_g = np.zeros((n_samples, n_elems), dtype=np.float32)
    voltage_residual = np.zeros((n_samples, n_meas), dtype=np.float32)
    coarse_residual_norm = np.zeros((n_samples,), dtype=np.float32)

    v_ref_abs = solver.V_uniform
    if np.isnan(v_ref_abs).any() or np.isinf(v_ref_abs).any():
        print("  [WARN] solver.V_uniform 非有限，传统反演绝对电压参考退回零向量")
        v_ref_abs = np.zeros(n_meas, dtype=np.float32)
    v_ref_abs = v_ref_abs.astype(np.float32)

    JT = J.T.astype(np.float32)
    for i in tqdm(range(n_samples), desc=f"precompute {split_name}"):
        V = voltages[i]
        V_diff = V[0] if V.ndim == 2 else V
        V_diff = V_diff.astype(np.float32)

        V_abs = V_diff + v_ref_abs
        sigma_i, _ = reconstructor.reconstruct(V_abs)
        sigma_i = sigma_i.astype(np.float32)

        V0 = J @ (sigma_i - sigma_ref)
        r = V_diff - V0.astype(np.float32)
        g = JT @ r
        g = (g - g.mean()) / (g.std() + 1e-6)

        sigma_0[i] = sigma_i
        voltage_residual[i] = r.astype(np.float32)
        physics_g[i] = g.astype(np.float32)
        coarse_residual_norm[i] = np.float32(
            np.linalg.norm(r) / (np.linalg.norm(V_diff) + 1e-8)
        )

    recreate_dataset(group, "sigma_0", sigma_0)
    recreate_dataset(group, "physics_g", physics_g)
    recreate_dataset(group, "voltage_residual", voltage_residual)
    recreate_dataset(group, "coarse_residual_norm", coarse_residual_norm, compression=None)
    print(f"  [{split_name}] 写入 sigma_0/physics_g/voltage_residual 完成")


def main():
    parser = argparse.ArgumentParser(description="Precompute residual EIT features")
    parser.add_argument("--h5", default="data/generated/eit_dataset.h5", help="HDF5 dataset path")
    parser.add_argument("--config", default="config/mesh_config.yaml", help="mesh config path")
    parser.add_argument("--jacobian", default="data/generated/jacobian.npy", help="Jacobian .npy path")
    parser.add_argument("--method", default="bp", choices=["bp", "jac"], help="traditional method")
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"], help="splits to process")
    parser.add_argument("--force", action="store_true", help="overwrite existing residual features")
    args = parser.parse_args()

    if not os.path.exists(args.h5):
        raise FileNotFoundError(args.h5)

    solver = EITForwardSolver(args.config)
    J = load_or_compute_jacobian(args.jacobian, solver)
    if J.shape != (solver.n_measurements, solver.n_elems):
        raise ValueError(
            f"Jacobian shape {J.shape} does not match solver "
            f"({solver.n_measurements}, {solver.n_elems})"
        )

    sigma_ref_value = solver.gt_cfg.get("conductivity_soil", 0.01)
    sigma_ref = np.full(solver.n_elems, sigma_ref_value, dtype=np.float32)
    reconstructor = build_reconstructor(solver, method=args.method)

    with h5py.File(args.h5, "a") as f:
        for split in args.splits:
            if split not in f:
                print(f"  [WARN] split 不存在，跳过: {split}")
                continue
            process_split(
                f[split],
                split,
                solver=solver,
                reconstructor=reconstructor,
                J=J,
                sigma_ref=sigma_ref,
                force=args.force,
            )


if __name__ == "__main__":
    main()
