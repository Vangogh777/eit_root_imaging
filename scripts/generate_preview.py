"""Generate dataset preview images."""
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri
from pyeit.mesh import create

def main():
    # Create mesh (same as used in data generation)
    print("Creating mesh...")
    mesh = create(n_el=16, h0=0.0025)
    pts = mesh.node
    triangles = mesh.element

    print(f"Mesh nodes: {pts.shape}")
    print(f"Mesh elements: {triangles.shape}")

    # Load dataset
    print("Loading dataset...")
    with h5py.File('data/generated/eit_dataset.h5', 'r') as f:
        train_sigmas = f['train/sigmas'][:]
        train_voltages = f['train/voltages'][:]

    print(f"Sigmas shape: {train_sigmas.shape}")
    print(f"Voltages shape: {train_voltages.shape}")
    print(f"Sigma range: [{train_sigmas.min():.4f}, {train_sigmas.max():.4f}]")

    # Generate preview for 4 samples
    print("Generating preview images...")
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle('New EIT Dataset Preview (Train Split - 1000 samples)', fontsize=14, fontweight='bold')

    for i in range(4):
        # Sigma plot
        ax_sigma = axes[0, i]
        sigma = train_sigmas[i]

        triang = tri.Triangulation(pts[:, 0], pts[:, 1], triangles)
        im = ax_sigma.tripcolor(triang, sigma, shading='flat', cmap='viridis')
        ax_sigma.set_aspect('equal')
        ax_sigma.set_title(f'Sample {i+1} σ', fontsize=11)
        ax_sigma.axis('off')
        plt.colorbar(im, ax=ax_sigma, fraction=0.046, pad=0.04)

        # Voltage plot
        ax_v = axes[1, i]
        voltage = train_voltages[i, 0]  # (1, 208) -> (208,)
        ax_v.plot(voltage, 'b-', linewidth=1)
        ax_v.set_title(f'Sample {i+1} Voltage', fontsize=11)
        ax_v.set_xlabel('Measurement')
        ax_v.set_ylabel('Voltage (V)')
        ax_v.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('results/dataset_preview/new_dataset_preview.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('Saved: results/dataset_preview/new_dataset_preview.png')

    # Create detailed single sample view
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('Sample 1 Detail View', fontsize=14, fontweight='bold')

    sigma = train_sigmas[0]
    voltage = train_voltages[0, 0]

    # Full sigma
    ax = axes[0]
    triang = tri.Triangulation(pts[:, 0], pts[:, 1], triangles)
    im = ax.tripcolor(triang, sigma, shading='flat', cmap='viridis')
    ax.set_aspect('equal')
    ax.set_title('Conductivity σ Distribution')
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Voltage bar
    ax = axes[1]
    ax.bar(range(len(voltage)), voltage, width=1.0, color='steelblue', alpha=0.7)
    ax.set_title('Boundary Voltage Measurements')
    ax.set_xlabel('Measurement Index')
    ax.set_ylabel('Voltage (V)')
    ax.grid(True, alpha=0.3, axis='y')

    # Voltage heatmap
    ax = axes[2]
    ax.imshow(voltage.reshape(1, -1), aspect='auto', cmap='RdBu')
    ax.set_title('Voltage Heatmap')
    ax.set_xlabel('Measurement Index')
    ax.set_yticks([])

    plt.tight_layout()
    plt.savefig('results/dataset_preview/new_dataset_detail.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('Saved: results/dataset_preview/new_dataset_detail.png')
    print('Done!')

if __name__ == '__main__':
    main()
