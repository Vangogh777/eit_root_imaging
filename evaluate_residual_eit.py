"""ResidualEIT validation & visualization script.
Loads best checkpoint, runs val set, saves metrics + sample images to results/.
"""

import os, sys, json
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))

from data.datasets.eit_dataset import MemoryEITDataset
from models.residual_eit import ResidualEIT


def _relative_error(pred, target):
    B = pred.shape[0]
    err = torch.norm(pred.view(B, -1) - target.view(B, -1), dim=1)
    norm = torch.norm(target.view(B, -1), dim=1)
    return (err / norm).mean().item()


def _correlation_coefficient(pred, target):
    B = pred.shape[0]
    pred_f = pred.view(B, -1)
    target_f = target.view(B, -1)
    pred_c = pred_f - pred_f.mean(dim=1, keepdim=True)
    target_c = target_f - target_f.mean(dim=1, keepdim=True)
    num = (pred_c * target_c).sum(dim=1)
    den = torch.sqrt((pred_c ** 2).sum(dim=1) * (target_c ** 2).sum(dim=1))
    cc = num / (den + 1e-8)
    return cc.mean().item()


def compute_per_sample_metrics(pred, target):
    """Return per-sample RE, CC arrays."""
    B = pred.shape[0]
    pred_f = pred.view(B, -1)
    target_f = target.view(B, -1)
    err = torch.norm(pred_f - target_f, dim=1)
    norm = torch.norm(target_f, dim=1)
    re = (err / norm).cpu().numpy()

    pred_c = pred_f - pred_f.mean(dim=1, keepdim=True)
    target_c = target_f - target_f.mean(dim=1, keepdim=True)
    num = (pred_c * target_c).sum(dim=1)
    den = torch.sqrt((pred_c ** 2).sum(dim=1) * (target_c ** 2).sum(dim=1))
    cc = (num / (den + 1e-8)).cpu().numpy()
    return re, cc


def render_mesh_panel(ax, mesh_nodes, mesh_elements, values, title,
                       cmap="viridis", vmin=None, vmax=None, with_cbar=True):
    """Render conductivity values on the 2D FEM triangular mesh (bucket cross-section)."""
    from matplotlib.tri import Triangulation
    triang = Triangulation(mesh_nodes[:, 0], mesh_nodes[:, 1], mesh_elements)
    im = ax.tripcolor(triang, facecolors=values, cmap=cmap,
                       vmin=vmin, vmax=vmax, shading="flat", edgecolors="none")
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=9, pad=3)
    if with_cbar:
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02)


def visualize_sample_row(ax_row, mesh_nodes, mesh_elements, target, pred, coarse, title,
                          vmin=0.05, vmax=0.25):
    """Plot 4 panels in a single row: target, pred, coarse, error on actual FEM mesh."""
    err = np.abs(pred - target)
    panels = [
        (target, "Ground Truth", "viridis", vmin, vmax),
        (pred, "Prediction", "viridis", vmin, vmax),
        (coarse, "Coarse (sigma_0)", "viridis", vmin, vmax),
        (err, "Absolute Error", "hot", 0, max(err.max(), 0.01)),
    ]
    for i, (data, lbl, cmap, lo, hi) in enumerate(panels):
        render_mesh_panel(ax_row[i], mesh_nodes, mesh_elements, data,
                          lbl, cmap=cmap, vmin=lo, vmax=hi, with_cbar=True)
    ax_row[0].set_ylabel(f"RE={title:.4f}" if isinstance(title, float) else title, fontsize=9)



def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── 1. Load checkpoint ──
    ckpt_path = "checkpoints/residual_eit_best.pt"
    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    print(f"Best RE: {ckpt['best_val_re']:.4f} at epoch {ckpt['best_epoch']}")

    # ── 2. Load validation dataset ──
    data_cfg = cfg["data"]
    h5_path = data_cfg["dataset_path"]
    jacobian_path = data_cfg.get("jacobian_path")
    model_cfg = cfg["model"]

    print(f"Loading dataset: {h5_path}")
    val_dataset = MemoryEITDataset(
        h5_path, split="val",
        voltage_mask_ratio=0.0,
        load_residual_features=True,
    )
    n_elems = val_dataset.n_elems
    n_freq = val_dataset.n_freq
    n_meas = val_dataset.n_meas
    centers = np.mean(val_dataset.mesh_nodes[val_dataset.mesh_elements], axis=1)
    elements = val_dataset.mesh_elements
    print(f"  n_elems={n_elems}, n_freq={n_freq}, n_meas={n_meas}")

    # ── 3. Load Jacobian ──
    J = None
    if jacobian_path and os.path.exists(jacobian_path):
        J = np.load(jacobian_path).astype(np.float32)
        print(f"  Jacobian loaded: {J.shape}")

    # ── 4. Instantiate model ──
    sigma_ref = cfg.get("physics", {}).get("sigma_ref", 0.1)
    print(f"Building ResidualEIT (hidden_dim={model_cfg['hidden_dim']}, gnn_layers={model_cfg['gnn_layers']})")
    model = ResidualEIT(
        n_frequencies=n_freq,
        n_meas=n_meas,
        n_elems=n_elems,
        hidden_dim=model_cfg.get("hidden_dim", 256),
        gnn_layers=model_cfg.get("gnn_layers", 4),
        dropout=model_cfg.get("dropout", 0.1),
        sigma_min=model_cfg.get("sigma_min", 0.005),
        sigma_max=model_cfg.get("sigma_max", 0.6),
        sigma_ref=sigma_ref,
        jacobian=J,
        delta_scale=model_cfg.get("delta_scale", 0.2),
        use_gat=model_cfg.get("use_gat", False),
        n_heads=model_cfg.get("n_heads", 4),
    )
    model.setup_mesh(centers, elements)

    # Load trained weights
    missing, unexpected = model.load_state_dict(ckpt["model_state_dict"], strict=False)
    if missing:
        print(f"  Missing keys: {missing}")
    if unexpected:
        print(f"  Unexpected keys: {unexpected}")
    model.to(device)
    model.eval()
    print(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # ── 5. Run inference ──
    print(f"\nRunning validation on {len(val_dataset)} samples...")
    preds, targets, coarse_list = [], [], []
    loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=64, shuffle=False, num_workers=4, pin_memory=True
    )

    with torch.no_grad():
        for batch in loader:
            voltages = batch["voltages"].to(device)
            sigma_0 = batch["sigma_0"].to(device)
            g = batch["physics_g"].to(device)
            residual = batch["voltage_residual"].to(device)

            out = model(voltages=voltages, sigma_0=sigma_0, g=g, residual=residual)

            preds.append(out["sigma"].cpu())
            targets.append(batch["sigmas"].cpu())
            coarse_list.append(sigma_0.cpu())

    pred = torch.cat(preds, dim=0)
    target = torch.cat(targets, dim=0)
    coarse = torch.cat(coarse_list, dim=0)

    # ── 6. Compute metrics ──
    re = _relative_error(pred, target)
    cc = _correlation_coefficient(pred, target)
    coarse_re = _relative_error(coarse, target)
    print(f"\n  ── Validation Metrics ──")
    print(f"  RE:        {re:.4f}")
    print(f"  CC:        {cc:.4f}")
    print(f"  Coarse RE: {coarse_re:.4f}")
    print(f"  RE improvement: {coarse_re - re:.4f} ({((coarse_re - re) / coarse_re * 100):.1f}%)")

    # Per-sample metrics for ranking
    re_per_sample, cc_per_sample = compute_per_sample_metrics(pred, target)
    sorted_idx = np.argsort(re_per_sample)

    # ── 7. Save metrics ──
    out_dir = "results/residual_eit_val"
    os.makedirs(out_dir, exist_ok=True)
    samples_dir = os.path.join(out_dir, "samples")
    os.makedirs(samples_dir, exist_ok=True)

    metrics = {
        "summary": {
            "RE": {"mean": re, "std": float(np.std(re_per_sample))},
            "CC": {"mean": cc, "std": float(np.std(cc_per_sample))},
            "Coarse RE": {"mean": coarse_re},
            "RE Improvement": {"mean": float(coarse_re - re)},
            "Best RE": {"mean": float(re_per_sample[sorted_idx[0]])},
            "Worst RE": {"mean": float(re_per_sample[sorted_idx[-1]])},
        },
        "num_samples": len(val_dataset),
        "best_epoch": ckpt["best_epoch"],
    }
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nMetrics saved to {out_dir}/metrics.json")

    # Get mesh data for visualization
    mesh_nodes = val_dataset.mesh_nodes
    mesh_elements = val_dataset.mesh_elements
    # Use only x,y for 2D (take first 2 columns if 3D)
    if mesh_nodes.shape[1] > 2:
        mesh_nodes_2d = mesh_nodes[:, :2]
    else:
        mesh_nodes_2d = mesh_nodes

    # ── 8. Shape-diverse sample selection ──
    # Use masks to estimate shape morphology
    val_masks = val_dataset.masks  # already loaded
    # Compute inclusion aspect ratio per sample
    centers_all = np.mean(val_dataset.mesh_nodes[val_dataset.mesh_elements], axis=1)
    if centers_all.shape[1] > 2:
        centers_all = centers_all[:, :2]
    aspects = []
    for i in range(len(val_masks)):
        inc = centers_all[val_masks[i].astype(bool)]
        if len(inc) > 1:
            sx, sy = inc[:, 0].std(), inc[:, 1].std()
            aspect = max(sx, sy) / (min(sx, sy) + 1e-8)
        else:
            aspect = 1.0
        aspects.append(aspect)
    aspects = np.array(aspects)

    # Classify and pick 2 best from each shape category
    cat_circle = np.where(aspects < 1.3)[0]        # circle-like
    cat_ellipse = np.where((aspects >= 1.3) & (aspects < 2.0))[0]  # ellipse-like
    cat_elong = np.where(aspects >= 2.0)[0]         # elongated / complex
    cats = {"circle": cat_circle, "ellipse": cat_ellipse, "elongated": cat_elong}
    diverse_idx = []
    for cname, cidx in cats.items():
        # Sort by RE within category, pick best 2
        cat_sorted = cidx[np.argsort(re_per_sample[cidx])]
        diverse_idx.extend(cat_sorted[:2])

    # ── 9. Generate sample visualizations ──
    n_viz = 8  # Show 8 best + 8 worst
    best_indices = sorted_idx[:n_viz]
    worst_indices = sorted_idx[-n_viz:][::-1]

    vmin = float(target.min().item())
    vmax = float(target.max().item())
    print(f"  Sigma range: [{vmin:.4f}, {vmax:.4f}]")

    # 8a. Best samples — rendered on FEM mesh
    fig, axes = plt.subplots(n_viz, 4, figsize=(16, 3 * n_viz))
    for i, idx in enumerate(best_indices):
        visualize_sample_row(
            axes[i], mesh_nodes_2d, mesh_elements,
            target[idx].numpy(), pred[idx].numpy(), coarse[idx].numpy(),
            title=re_per_sample[idx],
            vmin=vmin, vmax=vmax,
        )
    plt.tight_layout()
    best_path = os.path.join(out_dir, "best_8_samples.png")
    fig.savefig(best_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {best_path}")

    # 8b. Worst samples
    fig, axes = plt.subplots(n_viz, 4, figsize=(16, 3 * n_viz))
    for i, idx in enumerate(worst_indices):
        visualize_sample_row(
            axes[i], mesh_nodes_2d, mesh_elements,
            target[idx].numpy(), pred[idx].numpy(), coarse[idx].numpy(),
            title=re_per_sample[idx],
            vmin=vmin, vmax=vmax,
        )
    plt.tight_layout()
    worst_path = os.path.join(out_dir, "worst_8_samples.png")
    fig.savefig(worst_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {worst_path}")

    # 8c. Individual best/worst samples for lightbox — 2×2 grid on FEM mesh
    for label, indices in [("best", best_indices), ("worst", worst_indices)]:
        for rank, idx in enumerate(indices):
            fig, axs = plt.subplots(2, 2, figsize=(8, 7))
            axs_flat = axs.flatten()
            err = np.abs(pred[idx].numpy() - target[idx].numpy())
            panels = [
                (target[idx].numpy(), "Ground Truth", "viridis", vmin, vmax),
                (pred[idx].numpy(), "Prediction", "viridis", vmin, vmax),
                (coarse[idx].numpy(), "Coarse (sigma_0)", "viridis", vmin, vmax),
                (err, "Absolute Error", "hot", 0, max(err.max(), 0.01)),
            ]
            for j, (data, lbl, cmap, lo, hi) in enumerate(panels):
                render_mesh_panel(axs_flat[j], mesh_nodes_2d, mesh_elements, data,
                                  lbl, cmap=cmap, vmin=lo, vmax=hi, with_cbar=True)
            fig.suptitle(f"{label} #{rank+1}  RE={re_per_sample[idx]:.4f}  CC={cc_per_sample[idx]:.4f}",
                         fontsize=11, y=1.02)
            plt.tight_layout()
            fname = os.path.join(samples_dir, f"{label}_{rank+1}_RE{re_per_sample[idx]:.4f}.png")
            fig.savefig(fname, dpi=150, bbox_inches="tight")
            plt.close(fig)

    # 8d. RE vs CC scatter plot
    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(re_per_sample, cc_per_sample, c=re_per_sample,
                         cmap="viridis_r", alpha=0.6, s=20)
    ax.set_xlabel("Relative Error (RE)")
    ax.set_ylabel("Correlation Coefficient (CC)")
    ax.set_title(f"RE vs CC (n={len(re_per_sample)})\nMean RE={re:.4f}, Mean CC={cc:.4f}")
    plt.colorbar(scatter, ax=ax, label="RE")
    ax.grid(True, alpha=0.3)
    scatter_path = os.path.join(out_dir, "re_vs_cc.png")
    fig.savefig(scatter_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {scatter_path}")

    # 8e. RE histogram
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(re_per_sample, bins=30, alpha=0.7, color="steelblue", edgecolor="white")
    ax.axvline(re, color="red", linestyle="--", label=f"Mean RE={re:.4f}")
    ax.set_xlabel("Relative Error")
    ax.set_ylabel("Count")
    ax.set_title(f"RE Distribution (n={len(re_per_sample)})")
    ax.legend()
    ax.grid(True, alpha=0.3)
    hist_path = os.path.join(out_dir, "re_histogram.png")
    fig.savefig(hist_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {hist_path}")

    # 8f. Diverse shapes (best of each category)
    fig, axes = plt.subplots(len(diverse_idx), 4, figsize=(16, 3 * len(diverse_idx)))
    for i, idx in enumerate(diverse_idx):
        # Determine shape label
        a = aspects[idx]
        if a < 1.3:
            slabel = "circle"
        elif a < 2.0:
            slabel = "ellipse"
        else:
            slabel = "elongated/complex"
        visualize_sample_row(
            axes[i], mesh_nodes_2d, mesh_elements,
            target[idx].numpy(), pred[idx].numpy(), coarse[idx].numpy(),
            title=f"{slabel} RE={re_per_sample[idx]:.4f}",
            vmin=vmin, vmax=vmax,
        )
    plt.tight_layout()
    div_path = os.path.join(out_dir, "diverse_shapes.png")
    fig.savefig(div_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {div_path}")

    # ── 10. Report text ──
    report = (
        f"ResidualEIT Validation Report\n"
        f"{'=' * 40}\n"
        f"Best checkpoint from epoch {ckpt['best_epoch']}\n"
        f"Validation samples: {len(val_dataset)}\n\n"
        f"RE:  {re:.4f} ± {np.std(re_per_sample):.4f}\n"
        f"CC:  {cc:.4f} ± {np.std(cc_per_sample):.4f}\n"
        f"Coarse RE: {coarse_re:.4f}\n"
        f"RE improvement vs coarse: {(coarse_re - re):.4f} ({((coarse_re - re) / coarse_re * 100):.1f}%)\n\n"
        f"Model: ResidualEIT\n"
        f"Hidden dim: {model_cfg.get('hidden_dim', 256)}\n"
        f"GNN layers: {model_cfg.get('gnn_layers', 4)}\n"
        f"Parameters: {sum(p.numel() for p in model.parameters()):,}\n"
    )
    report_path = os.path.join(out_dir, "report.txt")
    with open(report_path, "w") as f:
        f.write(report)
    print(f"\n{report}")
    print(f"All results saved to {out_dir}/")
    print("Done!")


if __name__ == "__main__":
    main()
