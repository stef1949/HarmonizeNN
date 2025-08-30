#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Neural network-based batch correction for bulk RNA-seq.

Pipeline:
1) Read counts and metadata
2) Library-size normalize -> CPM -> log1p
3) Optional highly-variable gene (HVG) selection
4) Per-gene standardization (z-score) for training stability
5) Train autoencoder with gradient-reversal batch adversary
   (optional supervised head to preserve biological labels)
6) Export batch-corrected matrix in logCPM scale

Author: Steph Ritchie
License: MIT
"""

import argparse
import os
import random
from pathlib import Path
from typing import Optional, Tuple

import random
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import gridspec

import torch
import torch.nn as nn
# AMP GradScaler compatibility (torch>=2 provides torch.amp.GradScaler; older used torch.cuda.amp.GradScaler)
try:  # prefer new API to silence deprecation warnings when available
    from torch.amp import GradScaler  # type: ignore
    _NEW_GRADSCALER_API = True
except Exception:  # fallback
    from torch.cuda.amp import GradScaler  # type: ignore
    _NEW_GRADSCALER_API = False
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, accuracy_score
from sklearn.model_selection import train_test_split

# -------- Global perf toggles (safe no-ops if unsupported) --------
try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass

# ----------------------------
# Utils
# ----------------------------

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def to_device(x):
    return x.cuda() if torch.cuda.is_available() else x


def library_size_normalize(counts_df: pd.DataFrame, cpm_factor: float = 1e6) -> pd.DataFrame:
    """Counts (samples x genes) -> CPM -> log1p."""
    lib_sizes = counts_df.sum(axis=1).replace(0, np.nan)
    x = counts_df.div(lib_sizes, axis=0) * cpm_factor
    x = np.log1p(x)
    return x


def select_hvg_train_only(logcpm_df: pd.DataFrame, train_index: np.ndarray, n_hvg: int) -> pd.DataFrame:
    """Select HVGs using TRAIN variance only to avoid leakage; apply to all rows."""
    if n_hvg <= 0 or n_hvg >= logcpm_df.shape[1]:
        return logcpm_df
    vars_train = logcpm_df.iloc[train_index].var(axis=0)
    top = vars_train.nlargest(n_hvg).index
    return logcpm_df.loc[:, top]


def standardize_per_gene_fit_transform(
    logcpm_df: pd.DataFrame, train_index: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, StandardScaler]:
    """Fit scaler on TRAIN ONLY; transform both TRAIN and VAL via indices returned."""
    scaler = StandardScaler(with_mean=True, with_std=True)
    X_train = scaler.fit_transform(logcpm_df.values[train_index])
    X_all = scaler.transform(logcpm_df.values)  # for later inference
    return X_train, X_all, scaler


def inverse_standardize(X: np.ndarray, scaler: StandardScaler) -> np.ndarray:
    return scaler.inverse_transform(X)


# ----------------------------
# Data
# ----------------------------

class RNADataset(Dataset):
    def __init__(self, X: np.ndarray, batch_idx: np.ndarray,
                 label_idx: Optional[np.ndarray] = None):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.batch_idx = torch.tensor(batch_idx, dtype=torch.long)
        self.label_idx = torch.tensor(label_idx, dtype=torch.long) if label_idx is not None else None

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, i):
        if self.label_idx is None:
            return self.X[i], self.batch_idx[i]
        return self.X[i], self.batch_idx[i], self.label_idx[i]


def make_weighted_sampler(b_codes: np.ndarray, l_codes: Optional[np.ndarray] = None,
                          n_labels: Optional[int] = None) -> WeightedRandomSampler:
    """
    Balance sampling by batch or (batch,label) pairs.
    """
    if l_codes is not None and n_labels and n_labels > 1:
        # encode pair ids
        max_b = int(b_codes.max()) + 1
        pair = (b_codes.astype(int) * n_labels) + l_codes.astype(int)
        counts = np.bincount(pair, minlength=n_labels * max_b)
        weights = 1.0 / np.clip(counts[pair], 1, None)
    else:
        counts = np.bincount(b_codes, minlength=int(b_codes.max()) + 1)
        weights = 1.0 / np.clip(counts[b_codes], 1, None)
    weights = (weights / weights.mean()).astype(np.float64)
    return WeightedRandomSampler(weights.tolist(), num_samples=len(weights), replacement=True)


# ----------------------------
# Gradient Reversal
# ----------------------------

class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.lambda_, None


class GradReverseLayer(nn.Module):
    def __init__(self, lambda_: float = 1.0):
        super().__init__()
        self.lambda_ = lambda_

    def forward(self, x):
        return GradReverse.apply(x, self.lambda_)

    def set_lambda(self, lambda_: float):
        self.lambda_ = lambda_


# ----------------------------
# Model
# ----------------------------

def make_mlp(sizes, dropout=0.0, last_activation=None):
    """
    Hidden layers: Linear -> LayerNorm -> SiLU -> Dropout
    Output layer: optional activation per arg
    """
    layers = []
    for i in range(len(sizes) - 1):
        in_f, out_f = sizes[i], sizes[i + 1]
        layers.append(nn.Linear(in_f, out_f))
        if i < len(sizes) - 2:
            layers += [nn.LayerNorm(out_f), nn.SiLU(), nn.Dropout(dropout)]
        else:
            if last_activation == "relu":
                layers += [nn.ReLU()]
            elif last_activation == "tanh":
                layers += [nn.Tanh()]
            elif last_activation == "sigmoid":
                layers += [nn.Sigmoid()]
    return nn.Sequential(*layers)


class AEBatchCorrector(nn.Module):
    def __init__(
        self,
        n_genes: int,
        latent_dim: int = 32,
        enc_hidden=(1024, 256),
        dec_hidden=(256, 1024),
        adv_hidden=(128,),
        sup_hidden=(64,),
        n_batches: int = 2,
        n_labels: Optional[int] = None,
        dropout: float = 0.1,
        adv_lambda: float = 1.0,
    ):
        super().__init__()
        self.n_labels = n_labels
        self.grl = GradReverseLayer(lambda_=adv_lambda)

        enc_sizes = [n_genes] + list(enc_hidden) + [latent_dim]
        dec_sizes = [latent_dim] + list(dec_hidden) + [n_genes]
        self.encoder = make_mlp(enc_sizes, dropout=dropout, last_activation=None)
        self.decoder = make_mlp(dec_sizes, dropout=dropout, last_activation=None)

        adv_sizes = [latent_dim] + list(adv_hidden) + [n_batches]
        self.adv = make_mlp(adv_sizes, dropout=dropout, last_activation=None)

        if n_labels is not None:
            sup_sizes = [latent_dim] + list(sup_hidden) + [n_labels]
            self.sup = make_mlp(sup_sizes, dropout=dropout, last_activation=None)
        else:
            self.sup = None

    def forward(self, x, adv_lambda: Optional[float] = None):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        if adv_lambda is not None:
            self.grl.set_lambda(adv_lambda)
        z_rev = self.grl(z)
        batch_logits = self.adv(z_rev)
        label_logits = self.sup(z) if self.sup is not None else None
        return x_hat, batch_logits, label_logits, z

    @torch.no_grad()
    def reconstruct(self, x):
        """Fast path at inference time (skips adversary & GRL)."""
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat, z


# ----------------------------
# Training
# ----------------------------

def train_model(
    model: AEBatchCorrector,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 100,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    adv_weight: float = 1.0,
    sup_weight: float = 1.0,
    adv_lambda_schedule: str = "linear",  # or "constant", "sigmoid", "adaptive"
    recon_loss: str = "mse",
    use_amp: bool = False,
    scheduler_type: str = "none",
    grad_accum_steps: int = 1,
    adaptive_high_margin: float = 0.20,
    adaptive_low_margin: float = 0.05,
    adaptive_down_scale: float = 0.95,
    adaptive_up_scale: float = 1.05,
    patience: int = 15,
    early_stop_metric: str = "val_loss",  # choices: val_loss, val_batch_acc, val_sup_acc, objective_score
    log_latent_every: int = 1,  # 0 disables
    min_epochs: int = 0,
    early_stop_delta: float = 0.0,
    wandb_run: Optional[object] = None,
    batch_classes: Optional[list] = None,
) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # AdamW is generally nicer here
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Reconstruction loss
    recon_loss_fn = {"mse": nn.MSELoss(), "mae": nn.L1Loss(), "huber": nn.SmoothL1Loss()}[recon_loss]

    # Adversary cross-entropy: class-weighted + label smoothing
    # Try to infer class weights from the training dataset if available
    if batch_classes is not None and hasattr(train_loader, "dataset") and hasattr(train_loader.dataset, "batch_idx"):
        try:
            b_counts = np.bincount(train_loader.dataset.batch_idx.numpy(), minlength=len(batch_classes))
            w = (1.0 / np.clip(b_counts, 1, None))
            w = w / w.mean()
            batch_weights = torch.tensor(w, dtype=torch.float32, device=device)
        except Exception:
            batch_weights = torch.ones(len(batch_classes), dtype=torch.float32, device=device)
    else:
        batch_weights = None

    ce_batch = nn.CrossEntropyLoss(weight=batch_weights, label_smoothing=0.05)
    ce_sup = nn.CrossEntropyLoss(label_smoothing=0.05)

    # AMP setup
    amp_enabled = (use_amp and torch.cuda.is_available())
    amp_dtype = torch.bfloat16 if (amp_enabled and torch.cuda.is_bf16_supported()) else torch.float16
    # Use GradScaler without device_type kw (not present in some torch versions)
    scaler = GradScaler(enabled=amp_enabled)

    # LR scheduler
    if scheduler_type == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=5, min_lr=1e-6)
    elif scheduler_type == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    else:
        scheduler = None

    best_state = None
    best_val = None  # will set after first epoch depending on metric direction
    wait = 0

    # Adaptive lambda schedule state
    adaptive_lambda = adv_weight * 0.1
    # target adversary accuracy ~ random guessing
    n_batches = (len(batch_classes) if batch_classes else 2)
    target_acc = 1.0 / max(2, n_batches)

    def lambda_at_epoch(t, last_b_acc=None):
        nonlocal adaptive_lambda
        if adv_lambda_schedule == "constant":
            return adv_weight
        if adv_lambda_schedule == "linear":
            return adv_weight * min(1.0, (t + 1) / max(1, epochs // 3))
        if adv_lambda_schedule == "sigmoid":
            import math
            progress = (t + 1) / epochs
            return adv_weight * (1 / (1 + math.exp(-12 * (progress - 0.5))))
        if adv_lambda_schedule == "adaptive":
            if last_b_acc is not None:
                if last_b_acc > target_acc + adaptive_high_margin:
                    adaptive_lambda *= adaptive_down_scale
                elif last_b_acc < target_acc + adaptive_low_margin:
                    adaptive_lambda *= adaptive_up_scale
            adaptive_lambda = max(adv_weight * 0.05, min(adv_weight, adaptive_lambda))
            return adaptive_lambda
        raise ValueError("Unknown adv_lambda_schedule")

    history = {"train_loss": [], "val_loss": [], "val_batch_acc": [], "val_sup_acc": [], "objective_score": []}

    last_epoch_batch_acc = None
    for epoch in range(epochs):
        model.train()
        lam = lambda_at_epoch(epoch, last_b_acc=last_epoch_batch_acc)
        train_loss = 0.0
        opt.zero_grad(set_to_none=True)

        for step, batch in enumerate(train_loader):
            if len(batch) == 2:
                Xb, Bb = batch
                Lb = None
            else:
                Xb, Bb, Lb = batch

            Xb = Xb.to(device, non_blocking=True)
            Bb = Bb.to(device, non_blocking=True)
            Lb = Lb.to(device, non_blocking=True) if Lb is not None else None

            with torch.autocast(device_type='cuda', dtype=amp_dtype, enabled=amp_enabled):
                x_hat, b_logits, l_logits, _ = model(Xb, adv_lambda=lam)
                loss = recon_loss_fn(x_hat, Xb) + ce_batch(b_logits, Bb)
                if l_logits is not None and Lb is not None:
                    loss = loss + sup_weight * ce_sup(l_logits, Lb)

            # Gradient accumulation
            effective_loss = loss / max(1, grad_accum_steps)
            if amp_enabled:
                scaler.scale(effective_loss).backward()
            else:
                effective_loss.backward()

            is_update_step = ((step + 1) % grad_accum_steps == 0) or (step + 1 == len(train_loader))
            if is_update_step:
                if amp_enabled:
                    scaler.unscale_(opt)
                # Clip to stabilise GRL tug-of-war
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                if amp_enabled:
                    scaler.step(opt)
                    scaler.update()
                else:
                    opt.step()
                opt.zero_grad(set_to_none=True)

            train_loss += loss.item() * Xb.size(0)

        train_loss /= len(train_loader.dataset)

        # Validation
        model.eval()
        val_loss = 0.0
        all_b_true, all_b_pred = [], []
        all_l_true, all_l_pred = [], []
        all_z = []
        with torch.no_grad(), torch.autocast(device_type='cuda', dtype=amp_dtype, enabled=amp_enabled):
            for batch in val_loader:
                if len(batch) == 2:
                    Xb, Bb = batch
                    Lb = None
                else:
                    Xb, Bb, Lb = batch

                Xb = Xb.to(device, non_blocking=True)
                Bb = Bb.to(device, non_blocking=True)
                Lb = Lb.to(device, non_blocking=True) if Lb is not None else None

                x_hat, b_logits, l_logits, z = model(Xb, adv_lambda=lam)
                loss = recon_loss_fn(x_hat, Xb) + ce_batch(b_logits, Bb)
                if l_logits is not None and Lb is not None:
                    loss = loss + sup_weight * ce_sup(l_logits, Lb)
                val_loss += loss.item() * Xb.size(0)

                all_b_true.append(Bb.cpu().numpy())
                all_b_pred.append(b_logits.argmax(dim=1).cpu().numpy())
                if l_logits is not None and Lb is not None:
                    all_l_true.append(Lb.cpu().numpy())
                    all_l_pred.append(l_logits.argmax(dim=1).cpu().numpy())
                try:
                    all_z.append(z.cpu().numpy())
                except Exception:
                    pass

        val_loss /= len(val_loader.dataset)
        b_acc = accuracy_score(np.concatenate(all_b_true), np.concatenate(all_b_pred))
        last_epoch_batch_acc = b_acc
        if all_l_true:
            l_acc = accuracy_score(np.concatenate(all_l_true), np.concatenate(all_l_pred))
        else:
            l_acc = np.nan

        # Composite objective_score: maximise label accuracy, minimise val_loss, keep batch acc near target
        # target_acc defined above; penalize deviation
        deviation = abs(b_acc - target_acc)
        label_component = 0.0 if np.isnan(l_acc) else l_acc
        objective_score = label_component - val_loss - deviation

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_batch_acc"].append(b_acc)
        history["val_sup_acc"].append(l_acc)
        history["objective_score"].append(objective_score)

        current_lr = opt.param_groups[0]['lr']
        print(f"Epoch {epoch+1:03d}/{epochs} | "
              f"train {train_loss:.4f} | val {val_loss:.4f} | "
              f"val batch acc {b_acc:.3f}" + (f" | val label acc {l_acc:.3f}" if not np.isnan(l_acc) else ""))

        # W&B logging
        if wandb_run is not None:
            try:
                # include step so W&B associates metrics with epoch and shows history
                wandb_run.log({
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "val_batch_acc": b_acc,
                    "val_sup_acc": l_acc,
                    "adv_lambda": lam,
                    "lr": current_lr,
                    "objective_score": objective_score,
                }, step=epoch + 1)
            except Exception:
                pass

        # Scheduler step
        if scheduler is not None:
            if scheduler_type == "plateau":
                scheduler.step(val_loss)
            else:
                scheduler.step()

        # Early stopping logic (supports different metrics). For objective_score and *_acc we maximise; for losses we minimise.
        if early_stop_metric not in ("val_loss", "val_batch_acc", "val_sup_acc", "objective_score"):
            if wandb_run is not None:
                try:
                    import wandb  # noqa: F401
                    print(f"[WARN] Unknown early_stop_metric '{early_stop_metric}' - defaulting to val_loss")
                except Exception:
                    pass
            early_stop_metric_use = "val_loss"
        else:
            early_stop_metric_use = early_stop_metric

        current_metric_value = {
            "val_loss": val_loss,
            "val_batch_acc": b_acc,
            "val_sup_acc": l_acc,
            "objective_score": objective_score,
        }[early_stop_metric_use]

        maximize = early_stop_metric_use in ("val_batch_acc", "val_sup_acc", "objective_score")
        # Initialize best_val after first metric computed
        if best_val is None:
            best_val = current_metric_value
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            if maximize:
                diff = current_metric_value - best_val
                improved = diff > early_stop_delta
            else:
                diff = best_val - current_metric_value
                improved = diff > early_stop_delta
            if improved:
                best_val = current_metric_value
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                wait = 0
            else:
                # Only start counting patience AFTER min_epochs
                if (epoch + 1) >= max(1, min_epochs):
                    wait += 1
                if wait >= patience:
                    print(f"Early stopping at epoch {epoch+1} (no improvement > {early_stop_delta} for {patience} checks; best={best_val:.5f}).")
                    break

        # Latent-space visualization (UMAP if available, else PCA)
        if (
            wandb_run is not None
            and len(all_z) > 0
            and log_latent_every > 0
            and ((epoch + 1) % log_latent_every == 0)
        ):
            try:
                Z = np.concatenate(all_z)
                try:
                    import umap
                    reducer = umap.UMAP(n_components=2)
                    emb = reducer.fit_transform(Z)
                except Exception:
                    from sklearn.decomposition import PCA as _PCA
                    emb = _PCA(n_components=2).fit_transform(Z)

                codes = np.concatenate(all_b_true)
                fig = plt.figure(figsize=(5, 5))
                ax = fig.add_subplot(1, 1, 1)
                try:
                    import wandb
                    cmap = plt.get_cmap('tab10')
                    if batch_classes is not None:
                        unique_codes = np.unique(codes)
                        for c in unique_codes:
                            mask = codes == c
                            label = batch_classes[int(c)] if int(c) < len(batch_classes) else str(int(c))
                            color = cmap(int(c) % cmap.N)
                            ax.scatter(emb[mask, 0], emb[mask, 1], color=color, s=20, label=label, alpha=0.9)
                        ax.legend(loc='best', fontsize='small', markerscale=1.2)
                    else:
                        ax.scatter(emb[:, 0], emb[:, 1], c=codes, cmap='tab10', s=18)
                    ax.set_title(f"Epoch {epoch+1} latent")
                    ax.set_xticks([]); ax.set_yticks([])
                    # log image with step so each epoch image is stored in the run history
                    wandb_run.log({"latent_plot": wandb.Image(fig)}, step=epoch + 1)
                except Exception:
                    pass
                plt.close(fig)
            except Exception as e:
                print("[W&B] Warning: latent visualization failed:", e)

    if best_state is not None:
        model.load_state_dict(best_state)
    return {"model": model, "history": history}


# ----------------------------
# Evaluation helpers
# ----------------------------

def quick_silhouettes(z_latent: np.ndarray, batch_idx: np.ndarray, label_idx: Optional[np.ndarray]):
    out = {}
    try:
        zl = z_latent
        if zl.shape[1] < 2:
            zl = PCA(n_components=min(10, z_latent.shape[1])).fit_transform(z_latent)
        out["sil_batch"] = silhouette_score(zl, batch_idx, metric="euclidean")
        if label_idx is not None and len(np.unique(label_idx)) > 1:
            out["sil_label"] = silhouette_score(zl, label_idx, metric="euclidean")
        else:
            out["sil_label"] = np.nan
    except Exception as e:
        print("Silhouette error:", e)
        out["sil_batch"] = np.nan
        out["sil_label"] = np.nan
    return out


# ----------------------------
# Visualization helpers (integrated from visualise.py)
# ----------------------------

def viz_select_hvg(logcpm: pd.DataFrame, top_n: int) -> pd.DataFrame:
    """Select top-N most variable genes (columns) from a samples x genes logCPM matrix.
    If top_n <=0 or >= number of genes, returns input unchanged.
    """
    if top_n <= 0 or top_n >= logcpm.shape[1]:
        return logcpm
    vars_ = logcpm.var(axis=0)
    top = vars_.nlargest(top_n).index.tolist()
    return logcpm.loc[:, top]


def viz_plot_pca(
    logcpm: pd.DataFrame,
    meta: pd.DataFrame,
    batch_col: str,
    label_col: Optional[str],
    out_path: str,
    title: str = "PCA",
):
    """Generate a 2D PCA scatter plot colored by batch and (optionally) shaped by label.
    Saves figure to out_path.
    """
    try:
        pca = PCA(n_components=2)
        Z = pca.fit_transform(logcpm.values)
        df = pd.DataFrame(Z, index=logcpm.index, columns=["PC1", "PC2"])
        batches = meta[batch_col].astype("category")
        batch_codes = batches.cat.codes.loc[df.index].values
        batch_names = batches.cat.categories.tolist()

        fig, ax = plt.subplots(figsize=(6, 5))
        cmap = plt.get_cmap("tab10")
        for c in np.unique(batch_codes):
            mask = batch_codes == c
            label = batch_names[int(c)] if int(c) < len(batch_names) else str(int(c))
            ax.scatter(
                df.loc[mask, "PC1"],
                df.loc[mask, "PC2"],
                s=18,
                color=cmap(int(c) % cmap.N),
                label=label,
                alpha=0.85,
            )

        if label_col is not None and label_col in meta.columns:
            labels = meta.loc[df.index, label_col].astype("category")
            markers = ["o", "s", "^", "D", "v", "P", "X"]
            for i, lvl in enumerate(labels.cat.categories):
                mask = labels.cat.codes.values == i
                ax.scatter(
                    df.loc[mask, "PC1"],
                    df.loc[mask, "PC2"],
                    s=18,
                    facecolors="none",
                    edgecolors="k",
                    marker=markers[i % len(markers)],
                    linewidths=0.6,
                )

        ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
        ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
        ax.set_title(title)
        ax.legend(title="batch", bbox_to_anchor=(1.05, 1), loc="upper left", fontsize="small")
        fig.tight_layout()
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"[OK] Saved PCA: {out_path}")
    except Exception as e:
        print(f"[VIZ] Failed PCA ({out_path}): {e}")


def viz_plot_boxplots(
    logcpm_before: pd.DataFrame,
    logcpm_after: Optional[pd.DataFrame],
    meta: pd.DataFrame,
    batch_col: str,
    out_path: str,
):
    """Plot per-batch logCPM distributions (before vs after correction)."""
    try:
        batches = meta[batch_col].astype("category")
        batch_names = batches.cat.categories.tolist()
        before_groups = []
        after_groups = []
        labels = []
        for b in batch_names:
            samp_ids = meta.index[meta[batch_col] == b].tolist()
            if len(samp_ids) == 0:
                before_groups.append([])
                after_groups.append([])
                labels.append(b)
                continue
            vals_before = logcpm_before.loc[samp_ids].values.flatten()
            before_groups.append(vals_before)
            if logcpm_after is not None:
                vals_after = logcpm_after.loc[samp_ids].values.flatten()
                after_groups.append(vals_after)
            else:
                after_groups.append(None)
            labels.append(b)

        n = len(batch_names)
        fig = plt.figure(figsize=(max(6, n * 1.2), 5))
        gs = gridspec.GridSpec(1, 1)
        ax = fig.add_subplot(gs[0])

        positions = []
        data = []
        tick_pos = []
        width = 0.35
        for i in range(n):
            left = i * 2
            positions.append(left)
            data.append(before_groups[i])
            positions.append(left + width)
            data.append(after_groups[i] if after_groups[i] is not None else [])
            tick_pos.append(left + width / 2)

        bplots = ax.boxplot(
            data,
            positions=positions,
            widths=width,
            patch_artist=True,
            showfliers=False,
        )
        for idx, patch in enumerate(bplots["boxes"]):
            if idx % 2 == 0:
                patch.set_facecolor("#a6cee3")
            else:
                patch.set_facecolor("#b2df8a")

        ax.set_xticks(tick_pos)
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_ylabel("logCPM")
        ax.set_title("logCPM distributions by batch (before=blue, after=green)")
        fig.tight_layout()
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"[OK] Saved boxplots: {out_path}")
    except Exception as e:
        print(f"[VIZ] Failed boxplots ({out_path}): {e}")


# ----------------------------
# I/O + Main
# ----------------------------

def load_inputs(
    counts_path: Path,
    meta_path: Path,
    sample_col: str,
    batch_col: str,
    label_col: Optional[str],
    genes_in_rows: bool
):
    # Read counts (index is either sample names OR gene names depending on file layout)
    counts = pd.read_csv(counts_path, index_col=0)

    # If user explicitly declares genes are rows, transpose to samples x genes
    if genes_in_rows:
        counts = counts.T

    meta = pd.read_csv(meta_path)
    if sample_col not in meta.columns or batch_col not in meta.columns:
        raise ValueError(f"Metadata must contain '{sample_col}' and '{batch_col}' columns.")
    meta = meta.set_index(sample_col)

    if label_col is not None and label_col not in meta.columns:
        raise ValueError(f"Metadata missing label column '{label_col}'")

    # Straightforward intersection
    common = counts.index.intersection(meta.index)

    # Try auto-detect orientation if too few overlaps
    if len(common) < 2:
        cols_common = counts.columns.intersection(meta.index)
        if len(cols_common) >= 2:
            print("[INFO] Detected sample IDs in counts columns — transposing counts to samples x genes.")
            counts = counts.T
            common = counts.index.intersection(meta.index)

    # Normalize whitespace/quotes if still low overlap
    if len(common) < 2:
        try:
            counts.index = counts.index.astype(str).str.strip()
            counts.columns = counts.columns.astype(str).str.strip()
            meta.index = meta.index.astype(str).str.strip()
            common = counts.index.intersection(meta.index)
        except Exception:
            pass

    if len(common) < 2:
        raise ValueError("Too few overlapping samples between counts and metadata.")

    counts = counts.loc[common].copy()
    meta = meta.loc[common].copy()

    return counts, meta


def main():
    ap = argparse.ArgumentParser(description="Neural network bulk RNA-seq batch correction (adversarial autoencoder).")
    ap.add_argument("--counts", required=True, type=Path, help="Counts matrix CSV (genes x samples OR samples x genes). Index=feature or sample names.")
    ap.add_argument("--metadata", required=True, type=Path, help="Sample metadata CSV with at least [sample,batch].")
    ap.add_argument("--sample_col", default="sample", type=str, help="Column in metadata that has sample IDs.")
    ap.add_argument("--batch_col", default="batch", type=str, help="Column in metadata that has batch IDs.")
    ap.add_argument("--label_col", default=None, type=str, help="Optional biological label column (condition).")
    ap.add_argument("--genes_in_rows", action="store_true", help="Set if counts CSV is genes in rows, samples in columns (common layout).")
    ap.add_argument("--hvg", default=5000, type=int, help="Number of highly-variable genes to keep (0=all).")
    ap.add_argument("--epochs", default=200, type=int)
    ap.add_argument("--lr", default=1e-3, type=float)
    ap.add_argument("--batch_size", default=64, type=int)
    ap.add_argument("--latent_dim", default=32, type=int)
    ap.add_argument("--enc_hidden", default="1024,256", type=str)
    ap.add_argument("--dec_hidden", default="256,1024", type=str)
    ap.add_argument("--adv_hidden", default="128", type=str)
    ap.add_argument("--sup_hidden", default="64", type=str)
    ap.add_argument("--adv_weight", default=1.0, type=float, help="Strength of adversarial signal (also used as GRL lambda schedule target).")
    ap.add_argument("--sup_weight", default=1.0, type=float, help="Weight for supervised biology head.")
    ap.add_argument("--adv_lambda_schedule", default="linear", choices=["linear", "constant", "sigmoid", "adaptive"], help="Schedule for adversarial GRL lambda (adaptive balances adversary accuracy)")
    ap.add_argument("--adaptive_high_margin", default=0.20, type=float, help="Margin above target adversary accuracy to start reducing lambda (adaptive schedule)")
    ap.add_argument("--adaptive_low_margin", default=0.05, type=float, help="Margin above target adversary accuracy considered too low (increase lambda)")
    ap.add_argument("--adaptive_down_scale", default=0.95, type=float, help="Multiplicative factor to decrease lambda when adversary too strong")
    ap.add_argument("--adaptive_up_scale", default=1.05, type=float, help="Multipliclicative factor to increase lambda when adversary too weak")
    ap.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay (L2) for AdamW optimizer")
    ap.add_argument("--scheduler", default="none", choices=["none", "plateau", "cosine"], help="Learning rate scheduler")
    ap.add_argument("--recon_loss", default="mse", choices=["mse", "mae", "huber"], help="Reconstruction loss type")
    ap.add_argument("--amp", action="store_true", help="Enable mixed precision (AMP) training on CUDA")
    ap.add_argument("--dropout", default=0.1, type=float, help="Dropout probability for all MLPs")
    ap.add_argument("--grad_accum", default=1, type=int, help="Gradient accumulation steps")
    ap.add_argument("--num_workers", default=0, type=int, help="DataLoader num_workers")
    ap.add_argument("--pin_memory", action="store_true", help="Enable pin_memory in DataLoaders (CUDA only useful)")
    ap.add_argument("--patience", default=20, type=int)
    ap.add_argument("--min_epochs", default=0, type=int, help="Do not trigger early stopping before this many epochs (0=disable)")
    ap.add_argument("--early_stop_delta", default=0.0, type=float, help="Minimum improvement required to reset patience (direction depends on metric)")
    ap.add_argument("--seed", default=42, type=int)
    ap.add_argument("--compile", action="store_true", help="Use torch.compile (PyTorch 2+) for speed")
    ap.add_argument("--out_corrected", required=True, type=Path, help="Output CSV path for corrected matrix (logCPM scale).")
    ap.add_argument("--out_latent", default=None, type=Path, help="Optional CSV for latent embedding.")
    ap.add_argument("--save_model", default=None, type=Path, help="Optional path to save trained model .pt")
    ap.add_argument("--use_wandb", action="store_true", help="Log training and artifacts to Weights & Biases")
    ap.add_argument("--wandb_log", default="gradients", choices=["none", "gradients", "parameters", "all"], help="What to log with wandb.watch")
    ap.add_argument("--wandb_log_freq", default=100, type=int, help="How often (batches) to log gradients/params via wandb.watch")
    ap.add_argument("--expected_batches", default=None, type=int, help="Optional: assert the dataset contains exactly this many batches.")
    ap.add_argument("--label_values", default=None, type=str, help="Optional comma-separated expected label values that must appear in each batch (e.g. 'tumor,normal'). If not set, each batch must contain >=2 unique labels when --label_col is provided.")
    ap.add_argument("--early_stop_metric", default="val_loss", choices=["val_loss", "val_batch_acc", "val_sup_acc", "objective_score"], help="Metric to monitor for early stopping.")
    ap.add_argument("--log_latent_every", default=1, type=int, help="Log latent embedding plot every N epochs (0=disable).")
    # Visualization after training
    ap.add_argument("--generate_viz", action="store_true", help="Generate PCA and boxplot visualisations after training using the visualiser module")
    ap.add_argument("--viz_hvg_top", default=2000, type=int, help="Top-N most variable genes to use for PCA visualisations (0=use all)")
    ap.add_argument("--viz_pca_before", default="pca_before.png", type=str, help="Output path for PCA before correction")
    ap.add_argument("--viz_pca_after", default="pca_after.png", type=str, help="Output path for PCA after correction")
    ap.add_argument("--viz_boxplot", default="logCPM_boxplots.png", type=str, help="Output path for logCPM boxplots")
    args = ap.parse_args()


    set_seed(args.seed)

    # Optionally initialize Weights & Biases
    wandb_run = None
    if args.use_wandb:
        try:
            import wandb
            wandb_run = wandb.init(project="nn-batch-correction", config=vars(args))
        except Exception as e:
            print("[W&B] Warning: could not initialize wandb:", e)
            wandb_run = None

    # Load
    counts_raw, meta = load_inputs(
        args.counts, args.metadata, args.sample_col, args.batch_col, args.label_col, args.genes_in_rows
    )

    # Normalize -> logCPM (samples x genes)
    logcpm = library_size_normalize(counts_raw)

    # Encode batches + (optional) labels
    batch_cats = meta[args.batch_col].astype("category")
    batch_idx = batch_cats.cat.codes.values
    batch_classes = batch_cats.cat.categories.tolist()

    if args.label_col is not None:
        label_cats = meta[args.label_col].astype("category")
        label_idx = label_cats.cat.codes.values
        label_classes = label_cats.cat.categories.tolist()
    else:
        label_idx = None
        label_classes = None

    # Validate expected number of batches if requested
    if args.expected_batches is not None:
        if len(batch_classes) != args.expected_batches:
            raise ValueError(f"Expected {args.expected_batches} batches but found {len(batch_classes)} in metadata.")

    # Validate that each batch contains requested label values (or at least two labels)
    if args.label_col is not None:
        required_labels = None
        if args.label_values is not None:
            required_labels = [s.strip() for s in args.label_values.split(",") if s.strip()]
        for b in batch_classes:
            labels_in_batch = meta.loc[meta[args.batch_col] == b, args.label_col].dropna().unique().tolist()
            if required_labels is not None:
                missing = [r for r in required_labels if r not in labels_in_batch]
                if missing:
                    raise ValueError(f"Batch '{b}' missing required labels: {missing}")
            else:
                if len(labels_in_batch) < 2:
                    raise ValueError(f"Batch '{b}' contains fewer than 2 unique labels; found: {labels_in_batch}")

    # Train/Val split (stratify by batch, and label if present)
    strat = batch_idx if label_idx is None else (batch_idx.astype(str) + "|" + label_idx.astype(str))
    n_samples = logcpm.shape[0]
    n_batch_classes = len(np.unique(batch_idx))
    test_count = max(n_batch_classes, int(round(0.2 * n_samples)))

    train_ix, val_ix = train_test_split(
        np.arange(n_samples), test_size=test_count, random_state=args.seed, stratify=strat
    )

    # HVG selection on TRAIN ONLY (applied to all rows)
    logcpm = select_hvg_train_only(logcpm, train_ix, args.hvg)

    # Standardize per gene: fit on TRAIN ONLY; get TRAIN std and ALL std matrices
    X_train_std, X_all_std, scaler = standardize_per_gene_fit_transform(logcpm, train_ix)

    # Slice back out VAL std matrix using indices
    X_val_std = X_all_std[val_ix]

    b_train, b_val = batch_idx[train_ix], batch_idx[val_ix]
    if label_idx is not None:
        l_train, l_val = label_idx[train_ix], label_idx[val_ix]
    else:
        l_train = l_val = None

    # DataLoaders with weighted sampler for balanced mini-batches
    pin = args.pin_memory and torch.cuda.is_available()
    n_labels_for_pairs = (len(label_classes) if label_classes is not None else None)
    sampler = make_weighted_sampler(b_train, l_train, n_labels=n_labels_for_pairs)

    # Construct datasets
    ds_train = RNADataset(X_train_std, b_train, l_train)
    ds_val = RNADataset(X_val_std, b_val, l_val)

    # Build DataLoaders (prefetch only if workers>0)
    loader_kwargs = dict(
        drop_last=False,
        num_workers=args.num_workers,
        pin_memory=pin,
        persistent_workers=(args.num_workers > 0),
    )
    if args.num_workers > 0:
        loader_kwargs["prefetch_factor"] = 2

    dl_train = DataLoader(ds_train, batch_size=args.batch_size, sampler=sampler, **loader_kwargs)
    dl_val = DataLoader(ds_val, batch_size=args.batch_size, shuffle=False, **loader_kwargs)

    # Model
    enc_hidden = tuple(int(x) for x in args.enc_hidden.split(",") if x.strip())
    dec_hidden = tuple(int(x) for x in args.dec_hidden.split(",") if x.strip())
    adv_hidden = tuple(int(x) for x in args.adv_hidden.split(",") if x.strip())
    sup_hidden = tuple(int(x) for x in args.sup_hidden.split(",") if x.strip())

    model = AEBatchCorrector(
        n_genes=logcpm.shape[1],
        latent_dim=args.latent_dim,
        enc_hidden=enc_hidden,
        dec_hidden=dec_hidden,
        adv_hidden=adv_hidden,
        sup_hidden=sup_hidden,
        n_batches=len(batch_classes),
        n_labels=(len(label_classes) if label_classes is not None else None),
        dropout=args.dropout,
        adv_lambda=args.adv_weight,
    )

    # Optional torch.compile (PyTorch 2+)
    if args.compile:
        try:
            model = torch.compile(model, mode="max-autotune")
            print("[INFO] torch.compile enabled")
        except Exception as e:
            print(f"[WARN] torch.compile unavailable: {e}")

    # If wandb is enabled and a run exists, register model watching selectively
    if args.use_wandb and wandb_run is not None:
        try:
            import wandb
            if args.wandb_log != "none":
                log_choice = args.wandb_log if args.wandb_log in ("gradients", "parameters", "all") else None
                wandb.watch(model, log=log_choice, log_freq=args.wandb_log_freq)
                print(f"[W&B] Watching model (log={log_choice}, freq={args.wandb_log_freq})")
        except Exception as e:
            print("[W&B] Warning: wandb.watch failed:", e)

    # Train
    fit = train_model(
        model=model,
        train_loader=dl_train,
        val_loader=dl_val,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        adv_weight=args.adv_weight,
        sup_weight=args.sup_weight,
        adv_lambda_schedule=args.adv_lambda_schedule,
        recon_loss=args.recon_loss,
        use_amp=args.amp,
        scheduler_type=args.scheduler,
        grad_accum_steps=args.grad_accum,
        adaptive_high_margin=args.adaptive_high_margin,
        adaptive_low_margin=args.adaptive_low_margin,
        adaptive_down_scale=args.adaptive_down_scale,
        adaptive_up_scale=args.adaptive_up_scale,
        patience=args.patience,
    early_stop_metric=args.early_stop_metric,
    log_latent_every=args.log_latent_every,
    min_epochs=args.min_epochs,
    early_stop_delta=args.early_stop_delta,
        wandb_run=wandb_run,
        batch_classes=batch_classes,
    )
    model = fit["model"]

    # Inference: corrected = decoder(encoder(X)) using TRAIN-fitted scaler
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    with torch.no_grad():
        X_all = torch.tensor(X_all_std, dtype=torch.float32, device=device)
        x_hat, z = model.reconstruct(X_all)  # skip adversary at inference
        corr_std = x_hat.cpu().numpy()
        z_lat = z.cpu().numpy()

    # Inverse standardization -> logCPM scale
    corr_logcpm = inverse_standardize(corr_std, scaler)
    corrected_df = pd.DataFrame(corr_logcpm, index=logcpm.index, columns=logcpm.columns)
    corrected_df.to_csv(args.out_corrected)
    print(f"[OK] Wrote corrected matrix (logCPM) to: {args.out_corrected}")

    if args.out_latent:
        z_df = pd.DataFrame(z_lat, index=logcpm.index,
                            columns=[f"z{i+1}" for i in range(z_lat.shape[1])])
        z_df.to_csv(args.out_latent)
        print(f"[OK] Wrote latent embedding to: {args.out_latent}")

    if args.save_model:
        torch.save({"state_dict": model.state_dict(),
                    "batch_classes": batch_classes,
                    "label_classes": (label_classes if args.label_col is not None else None),
                    "genes": corrected_df.columns.tolist()},
                   args.save_model)
        print(f"[OK] Saved model to: {args.save_model}")
        if args.use_wandb and wandb_run is not None:
            try:
                import wandb
                artifact = wandb.Artifact("ae-model", type="model")
                artifact.add_file(str(args.save_model))
                wandb_run.log_artifact(artifact)
                print("[W&B] Uploaded model artifact")
            except Exception as e:
                print("[W&B] Warning: failed to upload model artifact:", e)

    # Quick diagnostics on latent space (uses VAL split implicitly through training)
    # Here we compute silhouettes on ALL latents against ALL labels for a quick sense.
    sil = quick_silhouettes(z_lat, batch_idx, (label_idx if args.label_col is not None else None))
    print(f"Silhouette (batch)  : {sil['sil_batch']:.3f}" if not np.isnan(sil['sil_batch']) else "Silhouette (batch): NA")
    print(f"Silhouette (label)  : {sil['sil_label']:.3f}" if not np.isnan(sil['sil_label']) else "Silhouette (label): NA")

    # Optional visualisations
    if args.generate_viz:
        try:
            # compute before-correction logCPM from raw counts
            logcpm_before = library_size_normalize(counts_raw)

            # select HVGs for viz (optionally different top-N)
            if args.viz_hvg_top and args.viz_hvg_top > 0:
                logcpm_before_sel = viz_select_hvg(logcpm_before, args.viz_hvg_top)
            else:
                logcpm_before_sel = logcpm_before

            # plot before
            viz_plot_pca(
                logcpm_before_sel,
                meta,
                args.batch_col,
                args.label_col if (args.label_col and args.label_col in meta.columns) else None,
                args.viz_pca_before,
                title=f"PCA (before) - HVG={args.viz_hvg_top}",
            )

            # load corrected -> ensure samples x genes orientation
            corrected_read = pd.read_csv(args.out_corrected, index_col=0)
            if corrected_read.shape[0] != meta.shape[0] and corrected_read.shape[1] == meta.shape[0]:
                corrected_read = corrected_read.T

            # If HVG was used for viz, intersect with corrected columns
            if args.viz_hvg_top and args.viz_hvg_top > 0:
                genes = [g for g in logcpm_before_sel.columns if g in corrected_read.columns]
                corrected_for_pca = corrected_read.loc[:, genes] if len(genes) >= 2 else corrected_read
            else:
                corrected_for_pca = corrected_read

            viz_plot_pca(
                corrected_for_pca,
                meta,
                args.batch_col,
                args.label_col if (args.label_col and args.label_col in meta.columns) else None,
                args.viz_pca_after,
                title=f"PCA (after) - HVG={args.viz_hvg_top}",
            )

            both_idx = logcpm_before.index.intersection(corrected_read.index)
            viz_plot_boxplots(
                logcpm_before.loc[both_idx],
                corrected_read.loc[both_idx],
                meta.loc[both_idx],
                args.batch_col,
                args.viz_boxplot,
            )
        except Exception as e:
            print(f"[VIZ] Failed to generate visualisations: {e}")


if __name__ == "__main__":
    main()