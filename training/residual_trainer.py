"""Training loop for the residual EIT route."""

from __future__ import annotations

import os
import sys
from collections import defaultdict
from typing import Dict, Optional

import numpy as np
import torch
import yaml
from tqdm import tqdm

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from data.datasets.eit_dataset import EITDataModule
from models.residual_eit import ResidualEIT
from training.loss import TVRegularizationLoss
from training.residual_loss import (
    RelativeMSELoss,
    ResidualMeasurementConsistencyLoss,
    ResidualSparsityLoss,
    ResidualSmoothnessLoss,
    weighted_residual_loss,
)


class ResidualEITTrainer:
    """Trainer for ResidualEIT."""

    def __init__(self, config_path: str = "config/residual_eit_config.yaml"):
        with open(config_path, "r", encoding="utf-8") as f:
            self.cfg = yaml.safe_load(f)
        self.config_path = config_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.dm = None
        self.optimizer = None
        self.scheduler = None
        self.loss_fns = {}
        self.loss_weights = self.cfg["training"]["loss_weights"]
        self.best_val_re = float("inf")

    def setup(self, model: Optional[ResidualEIT] = None, datamodule: Optional[EITDataModule] = None):
        data_cfg = self.cfg["data"]
        train_cfg = self.cfg["training"]
        model_cfg = self.cfg["model"]

        if datamodule is None:
            self.dm = EITDataModule(
                h5_path=data_cfg["dataset_path"],
                batch_size=train_cfg["batch_size"],
                val_batch_size=train_cfg.get("val_batch_size"),
                num_workers=data_cfg.get("num_workers", 4),
                prefetch_factor=data_cfg.get("prefetch_factor", 2),
                voltage_mask_ratio=train_cfg.get("voltage_mask_ratio", 0.0),
                jacobian_path=data_cfg.get("jacobian_path"),
                load_residual_features=True,
            )
        else:
            self.dm = datamodule

        ds = self.dm.train_dataset
        n_elems = ds.n_elems
        n_freq = ds.n_freq
        n_meas = ds.n_meas
        centers = np.mean(ds.mesh_nodes[ds.mesh_elements], axis=1)
        elements = ds.mesh_elements

        J = self._load_jacobian(data_cfg.get("jacobian_path"))
        sigma_ref = self.cfg.get("physics", {}).get("sigma_ref", 0.01)

        if model is None:
            self.model = ResidualEIT(
                n_frequencies=n_freq,
                n_meas=n_meas,
                n_elems=n_elems,
                hidden_dim=model_cfg.get("hidden_dim", 256),
                gnn_layers=model_cfg.get("gnn_layers", 4),
                dropout=model_cfg.get("dropout", 0.1),
                sigma_min=model_cfg.get("sigma_min", 0.005),
                sigma_max=model_cfg.get("sigma_max", 0.1),
                sigma_ref=sigma_ref,
                jacobian=J,
                delta_scale=model_cfg.get("delta_scale", 0.02),
                use_gat=model_cfg.get("use_gat", True),
                n_heads=model_cfg.get("n_heads", 4),
            )
        else:
            self.model = model

        self.model.setup_mesh(centers, elements)
        self.model.to(self.device)

        self.loss_fns = {
            "supervised": RelativeMSELoss(),
            "residual_measurement": ResidualMeasurementConsistencyLoss(J),
            "tv": TVRegularizationLoss(
                element_centers=torch.from_numpy(centers).float(),
                mesh_elements=torch.from_numpy(elements).long(),
                mesh_nodes=torch.from_numpy(ds.mesh_nodes).float(),
            ),
            "delta_l1": ResidualSparsityLoss(),
            "delta_smooth": ResidualSmoothnessLoss(self.model._edge_idx),
        }
        for loss_fn in self.loss_fns.values():
            if hasattr(loss_fn, "to"):
                loss_fn.to(self.device)

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=train_cfg.get("learning_rate", 3e-4),
            weight_decay=train_cfg.get("weight_decay", 1e-5),
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=train_cfg["epochs"],
            eta_min=train_cfg.get("eta_min", 1e-6),
        )

        params = sum(p.numel() for p in self.model.parameters())
        print(f"[ResidualEITTrainer] device={self.device}, params={params:,}")

    def train(self):
        train_cfg = self.cfg["training"]
        output_cfg = self.cfg.get("output", {})
        epochs = train_cfg["epochs"]
        save_path = output_cfg.get("checkpoint_path", "checkpoints/residual_eit_best.pt")

        train_loader = self.dm.train_dataloader()
        val_loader = self.dm.val_dataloader()
        best_state = None
        best_epoch = 0

        for epoch in range(1, epochs + 1):
            train_metrics = self._run_epoch(train_loader, train=True, epoch=epoch)
            val_metrics = self._run_epoch(val_loader, train=False, epoch=epoch)
            self.scheduler.step()

            lr = self.optimizer.param_groups[0]["lr"]
            print(
                f"Epoch {epoch:03d}/{epochs} | "
                f"train_loss={train_metrics['loss']:.6f} | "
                f"val_loss={val_metrics['loss']:.6f} | "
                f"val_RE={val_metrics['re']:.4f} | "
                f"val_coarse_RE={val_metrics['coarse_re']:.4f} | "
                f"lr={lr:.2e}"
            )

            if val_metrics["re"] < self.best_val_re:
                self.best_val_re = val_metrics["re"]
                best_epoch = epoch
                best_state = {
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "config": self.cfg,
                    "best_val_re": self.best_val_re,
                    "best_epoch": best_epoch,
                }
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                torch.save(best_state, save_path)

        print(f"[ResidualEITTrainer] best val RE={self.best_val_re:.4f} at epoch {best_epoch}")
        return best_state

    def _run_epoch(self, loader, train: bool, epoch: int) -> Dict[str, float]:
        self.model.train(train)
        metrics = defaultdict(float)
        all_pred = []
        all_target = []
        all_coarse = []

        pbar = tqdm(loader, desc=("train" if train else "val") + f" {epoch}", leave=False)
        for batch in pbar:
            voltages = batch["voltages"].to(self.device)
            target = batch["sigmas"].to(self.device)
            sigma_0 = batch["sigma_0"].to(self.device)
            g = batch["physics_g"].to(self.device)
            residual = batch["voltage_residual"].to(self.device)

            with torch.set_grad_enabled(train):
                out = self.model(
                    voltages=voltages,
                    sigma_0=sigma_0,
                    g=g,
                    residual=residual,
                )
                losses = {
                    "supervised": self.loss_fns["supervised"](out["sigma"], target),
                    "residual_measurement": self.loss_fns["residual_measurement"](
                        out["delta_sigma"], residual),
                    "tv": self.loss_fns["tv"](out["sigma"]),
                    "delta_l1": self.loss_fns["delta_l1"](out["delta_sigma"]),
                    "delta_smooth": self.loss_fns["delta_smooth"](out["delta_sigma"]),
                }
                total = weighted_residual_loss(losses, self.loss_weights)

                if train:
                    self.optimizer.zero_grad()
                    total.backward()
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.cfg["training"].get("grad_clip", 1.0),
                    )
                    self.optimizer.step()

            batch_size = voltages.shape[0]
            metrics["loss"] += total.item() * batch_size
            for name, loss in losses.items():
                metrics[name] += loss.item() * batch_size
            metrics["n"] += batch_size

            all_pred.append(out["sigma"].detach().cpu())
            all_target.append(target.detach().cpu())
            all_coarse.append(sigma_0.detach().cpu())
            pbar.set_postfix({"loss": f"{total.item():.4f}"})

        n = max(metrics.pop("n"), 1)
        avg = {k: v / n for k, v in metrics.items()}
        pred = torch.cat(all_pred, dim=0)
        target = torch.cat(all_target, dim=0)
        coarse = torch.cat(all_coarse, dim=0)
        avg["re"] = self._relative_error(pred, target)
        avg["coarse_re"] = self._relative_error(coarse, target)
        avg["cc"] = self._correlation_coefficient(pred, target)
        return avg

    def _load_jacobian(self, path: Optional[str]) -> torch.Tensor:
        if not path or not os.path.exists(path):
            raise FileNotFoundError(
                f"Residual training requires a precomputed Jacobian, got: {path}"
            )
        J = np.load(path).astype(np.float32)
        if J.ndim == 3:
            J = J[0]
        return torch.from_numpy(J).float().to(self.device)

    @staticmethod
    def _relative_error(pred: torch.Tensor, target: torch.Tensor) -> float:
        return (
            torch.norm(pred - target, dim=-1) /
            (torch.norm(target, dim=-1) + 1e-8)
        ).mean().item()

    @staticmethod
    def _correlation_coefficient(pred: torch.Tensor, target: torch.Tensor) -> float:
        pred_c = pred - pred.mean(dim=-1, keepdim=True)
        target_c = target - target.mean(dim=-1, keepdim=True)
        cov = (pred_c * target_c).sum(dim=-1)
        pred_std = torch.sqrt((pred_c ** 2).sum(dim=-1) + 1e-8)
        target_std = torch.sqrt((target_c ** 2).sum(dim=-1) + 1e-8)
        return (cov / (pred_std * target_std + 1e-8)).mean().item()
