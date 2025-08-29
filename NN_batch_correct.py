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

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, accuracy_score
from sklearn.model_selection import train_test_split


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


def select_hvg(logcpm_df: pd.DataFrame, n_hvg: int) -> pd.DataFrame:
    if n_hvg <= 0 or n_hvg >= logcpm_df.shape[1]:
        return logcpm_df
    vars_ = logcpm_df.var(axis=0)
    top = vars_.nlargest(n_hvg).index
    return logcpm_df.loc[:, top]


def standardize_per_gene(logcpm_df: pd.DataFrame) -> Tuple[pd.DataFrame, StandardScaler]:
    scaler = StandardScaler(with_mean=True, with_std=True)
    X = scaler.fit_transform(logcpm_df.values)  # samples x genes
    return pd.DataFrame(X, index=logcpm_df.index, columns=logcpm_df.columns), scaler


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
    layers = []
    for i in range(len(sizes) - 1):
        in_f, out_f = sizes[i], sizes[i+1]
        layers += [nn.Linear(in_f, out_f)]
        if i < len(sizes) - 2:  # hidden
            layers += [nn.ReLU(), nn.Dropout(dropout)]
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
    adv_lambda_schedule: str = "linear",  # or "constant"
    recon_loss: str = "mse",
    use_amp: bool = False,
    scheduler_type: str = "none",
    grad_accum_steps: int = 1,
    adaptive_high_margin: float = 0.20,
    adaptive_low_margin: float = 0.05,
    adaptive_down_scale: float = 0.95,
    adaptive_up_scale: float = 1.05,
    patience: int = 15,
    wandb_run: Optional[object] = None,
    save_best_path: Optional[Path] = None,
    batch_classes: Optional[list] = None,
) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    if recon_loss == "mse":
        recon_loss_fn = nn.MSELoss()
    elif recon_loss == "mae":
        recon_loss_fn = nn.L1Loss()
    elif recon_loss == "huber":
        recon_loss_fn = nn.SmoothL1Loss()
    else:
        raise ValueError("Unsupported recon_loss; choose from mse, mae, huber")
    ce = nn.CrossEntropyLoss()
    scaler = GradScaler(enabled=use_amp and torch.cuda.is_available())

    if scheduler_type == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=5, min_lr=1e-6)
    elif scheduler_type == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    else:
        scheduler = None

    best_state = None
    best_val = float("inf")
    wait = 0

    # For adaptive schedule we maintain stateful lambda that reacts to adversary accuracy
    adaptive_lambda = adv_weight * 0.1
    target_acc = 1.0 / max(2, (len(batch_classes) if batch_classes else 2))  # near random guessing
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
            # If adversary accuracy much higher than target, reduce lambda to avoid overpowering encoder.
            # If too low (encoder already confusing), gently raise lambda.
            if last_b_acc is not None:
                if last_b_acc > target_acc + adaptive_high_margin:
                    adaptive_lambda *= adaptive_down_scale
                elif last_b_acc < target_acc + adaptive_low_margin:
                    adaptive_lambda *= adaptive_up_scale
            adaptive_lambda = max(adv_weight * 0.05, min(adv_weight, adaptive_lambda))
            return adaptive_lambda
        raise ValueError("Unknown adv_lambda_schedule")

    history = {"train_loss": [], "val_loss": [], "val_batch_acc": [], "val_sup_acc": []}

    last_epoch_batch_acc = None
    for epoch in range(epochs):
        model.train()
        lam = lambda_at_epoch(epoch, last_b_acc=last_epoch_batch_acc)
        train_loss = 0.0
        opt.zero_grad()
        for step, batch in enumerate(train_loader):
            if len(batch) == 2:
                Xb, Bb = batch
                Lb = None
            else:
                Xb, Bb, Lb = batch

            Xb = Xb.to(device)
            Bb = Bb.to(device)
            Lb = Lb.to(device) if Lb is not None else None

            with torch.amp.autocast(device_type='cuda', enabled=use_amp and torch.cuda.is_available()):
                x_hat, b_logits, l_logits, _ = model(Xb, adv_lambda=lam)
                loss = recon_loss_fn(x_hat, Xb) + ce(b_logits, Bb)
                if l_logits is not None and Lb is not None:
                    loss = loss + sup_weight * ce(l_logits, Lb)

            # Gradient accumulation
            effective_loss = loss / max(1, grad_accum_steps)
            if use_amp and torch.cuda.is_available():
                scaler.scale(effective_loss).backward()
            else:
                effective_loss.backward()

            is_update_step = ((step + 1) % grad_accum_steps == 0) or (step + 1 == len(train_loader))
            if is_update_step:
                if use_amp and torch.cuda.is_available():
                    scaler.step(opt)
                    scaler.update()
                else:
                    opt.step()
                opt.zero_grad()
            train_loss += loss.item() * Xb.size(0)

        train_loss /= len(train_loader.dataset)

        # validation
        model.eval()
        val_loss = 0.0
        all_b_true, all_b_pred = [], []
        all_l_true, all_l_pred = [], []
        all_z = []
        with torch.no_grad():
            for batch in val_loader:
                if len(batch) == 2:
                    Xb, Bb = batch
                    Lb = None
                else:
                    Xb, Bb, Lb = batch
                Xb = Xb.to(device)
                Bb = Bb.to(device)
                Lb = Lb.to(device) if Lb is not None else None

                with torch.amp.autocast(device_type='cuda', enabled=use_amp and torch.cuda.is_available()):
                    x_hat, b_logits, l_logits, z = model(Xb, adv_lambda=lam)
                    loss = recon_loss_fn(x_hat, Xb) + ce(b_logits, Bb)
                    if l_logits is not None and Lb is not None:
                        loss = loss + sup_weight * ce(l_logits, Lb)
                val_loss += loss.item() * Xb.size(0)

                all_b_true.append(Bb.cpu().numpy())
                all_b_pred.append(b_logits.argmax(dim=1).cpu().numpy())

                if l_logits is not None and Lb is not None:
                    all_l_true.append(Lb.cpu().numpy())
                    all_l_pred.append(l_logits.argmax(dim=1).cpu().numpy())
                # collect latent vectors for visualization
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

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_batch_acc"].append(b_acc)
        history["val_sup_acc"].append(l_acc)

        print(f"Epoch {epoch+1:03d}/{epochs} | "
              f"train {train_loss:.4f} | val {val_loss:.4f} | "
              f"val batch acc {b_acc:.3f}" + (f" | val label acc {l_acc:.3f}" if not np.isnan(l_acc) else ""))

        # Log to Weights & Biases if enabled
        if wandb_run is not None:
            try:
                wandb_run.log({
                    "epoch": epoch + 1,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "val_batch_acc": b_acc,
                    "val_sup_acc": l_acc,
                    "adv_lambda": lam,
                    "lr": current_lr,
                })
            except Exception:
                pass

        # wandb logging handled via wandb_run above

        # Scheduler step (plateau needs val loss)
        if scheduler is not None:
            if scheduler_type == "plateau":
                scheduler.step(val_loss)
            else:
                scheduler.step()
        current_lr = opt.param_groups[0]['lr']

        # Early stopping on validation loss
        if val_loss < best_val - 1e-5:
            best_val = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
            # save best model to disk if requested
            if save_best_path is not None:
                try:
                    torch.save({"state_dict": model.state_dict()}, str(save_best_path))
                    print(f"[OK] Saved best model to: {save_best_path}")
                    if wandb_run is not None:
                        try:
                            import wandb
                            art = wandb.Artifact("ae-best-model", type="model")
                            art.add_file(str(save_best_path))
                            wandb_run.log_artifact(art)
                            print("[W&B] Uploaded best-model artifact")
                        except Exception as e:
                            print("[W&B] Warning: failed to upload best-model artifact:", e)
                except Exception as e:
                    print("Warning: failed to save best model:", e)
        else:
            wait += 1
            if wait >= patience:
                print("Early stopping.")
                break

        # Latent-space visualization (UMAP if available, else PCA)
        if wandb_run is not None and len(all_z) > 0:
            try:
                Z = np.concatenate(all_z)
                try:
                    import umap
                    reducer = umap.UMAP(n_components=2)
                    emb = reducer.fit_transform(Z)
                except Exception:
                    from sklearn.decomposition import PCA as _PCA
                    emb = _PCA(n_components=2).fit_transform(Z)

                fig = plt.figure(figsize=(5, 5))
                ax = fig.add_subplot(1, 1, 1)
                codes = np.concatenate(all_b_true)
                try:
                    # color by batch label names when provided, otherwise by numeric code
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
                except Exception:
                    ax.scatter(emb[:, 0], emb[:, 1], c=codes, cmap='tab10', s=18)
                ax.set_title(f"Epoch {epoch+1} latent")
                ax.set_xticks([])
                ax.set_yticks([])
                try:
                    import wandb
                    wandb_run.log({"latent_plot": wandb.Image(fig), "epoch": epoch + 1})
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
        # Use latent directly; if degenerate, fallback to PCA(10)
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

    # First, try a straightforward intersection between counts index and metadata sample IDs
    common = counts.index.intersection(meta.index)

    # If very few overlaps, attempt to auto-detect orientation: maybe samples are stored in columns
    if len(common) < 2:
        # If samples appear in columns, their names will intersect with metadata index
        cols_common = counts.columns.intersection(meta.index)
        if len(cols_common) >= 2:
            print("[INFO] Detected sample IDs in counts columns — transposing counts to samples x genes.")
            counts = counts.T
            common = counts.index.intersection(meta.index)

    # As a last attempt, normalize simple whitespace/quote differences and retry
    if len(common) < 2:
        try:
            counts_index_str = counts.index.astype(str).str.strip()
            counts.columns = counts.columns.astype(str).str.strip()
            meta_index_str = meta.index.astype(str).str.strip()
            # assign back the stripped indices for matching
            counts.index = counts_index_str
            meta.index = meta_index_str
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
    ap.add_argument("--adaptive_up_scale", default=1.05, type=float, help="Multiplicative factor to increase lambda when adversary too weak")
    ap.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay (L2) for Adam optimizer")
    ap.add_argument("--scheduler", default="none", choices=["none", "plateau", "cosine"], help="Learning rate scheduler")
    ap.add_argument("--recon_loss", default="mse", choices=["mse", "mae", "huber"], help="Reconstruction loss type")
    ap.add_argument("--amp", action="store_true", help="Enable mixed precision (AMP) training on CUDA")
    ap.add_argument("--dropout", default=0.1, type=float, help="Dropout probability for all MLPs")
    ap.add_argument("--grad_accum", default=1, type=int, help="Gradient accumulation steps")
    ap.add_argument("--num_workers", default=0, type=int, help="DataLoader num_workers")
    ap.add_argument("--pin_memory", action="store_true", help="Enable pin_memory in DataLoaders (CUDA only useful)")
    ap.add_argument("--patience", default=20, type=int)
    ap.add_argument("--seed", default=42, type=int)
    ap.add_argument("--out_corrected", required=True, type=Path, help="Output CSV path for corrected matrix (logCPM scale).")
    ap.add_argument("--out_latent", default=None, type=Path, help="Optional CSV for latent embedding.")
    ap.add_argument("--save_model", default=None, type=Path, help="Optional path to save trained model .pt")
    ap.add_argument("--use_wandb", action="store_true", help="Log training and artifacts to Weights & Biases")
    ap.add_argument("--wandb_log", default="gradients", choices=["none", "gradients", "parameters", "all"], help="What to log with wandb.watch")
    ap.add_argument("--wandb_log_freq", default=100, type=int, help="How often (batches) to log gradients/params via wandb.watch")
    ap.add_argument("--expected_batches", default=None, type=int, help="Optional: assert the dataset contains exactly this many batches.")
    ap.add_argument("--label_values", default=None, type=str, help="Optional comma-separated expected label values that must appear in each batch (e.g. 'tumor,normal'). If not set, each batch must contain >=2 unique labels when --label_col is provided.")
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
            # allow user to set WANDB_MODE=offline for local-only runs
            wandb_run = wandb.init(project="nn-batch-correction", config=vars(args))
        except Exception as e:
            print("[W&B] Warning: could not initialize wandb:", e)
            wandb_run = None

    if args.use_wandb:
        import wandb
        wandb.init(project="nn-batch-correction", config=vars(args))

    # Load
    counts_raw, meta = load_inputs(
        args.counts, args.metadata, args.sample_col, args.batch_col, args.label_col, args.genes_in_rows
    )

    # Normalize -> logCPM
    logcpm = library_size_normalize(counts_raw)
    # HVGs
    logcpm = select_hvg(logcpm, args.hvg)

    # Standardize per gene for training
    logcpm_std, scaler = standardize_per_gene(logcpm)

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
        # group metadata by batch and check
        for b in batch_classes:
            labels_in_batch = meta.loc[meta[args.batch_col] == b, args.label_col].dropna().unique().tolist()
            if required_labels is not None:
                missing = [r for r in required_labels if r not in labels_in_batch]
                if missing:
                    raise ValueError(f"Batch '{b}' missing required labels: {missing}")
            else:
                if len(labels_in_batch) < 2:
                    raise ValueError(f"Batch '{b}' contains fewer than 2 unique labels; found: {labels_in_batch}")

    # Train/Val split (stratify by batch)
    strat = batch_idx
    n_samples = logcpm_std.shape[0]
    # choose an integer test size that's at least 1 and at least the number of batch classes
    n_batch_classes = len(np.unique(batch_idx))
    test_count = max(1, int(round(0.2 * n_samples)))
    if test_count < n_batch_classes:
        test_count = n_batch_classes
    train_ix, val_ix = train_test_split(
        np.arange(n_samples), test_size=test_count, random_state=args.seed, stratify=strat
    )

    X_train = logcpm_std.values[train_ix]
    X_val = logcpm_std.values[val_ix]
    b_train, b_val = batch_idx[train_ix], batch_idx[val_ix]
    if label_idx is not None:
        l_train, l_val = label_idx[train_ix], label_idx[val_ix]
    else:
        l_train = l_val = None

    ds_train = RNADataset(X_train, b_train, l_train)
    ds_val = RNADataset(X_val, b_val, l_val)

    dl_train = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True, drop_last=False,
                          num_workers=args.num_workers, pin_memory=args.pin_memory)
    dl_val = DataLoader(ds_val, batch_size=args.batch_size, shuffle=False, drop_last=False,
                        num_workers=args.num_workers, pin_memory=args.pin_memory)

    # Model
    enc_hidden = tuple(int(x) for x in args.enc_hidden.split(",") if x.strip())
    dec_hidden = tuple(int(x) for x in args.dec_hidden.split(",") if x.strip())
    adv_hidden = tuple(int(x) for x in args.adv_hidden.split(",") if x.strip())
    sup_hidden = tuple(int(x) for x in args.sup_hidden.split(",") if x.strip())

    model = AEBatchCorrector(
        n_genes=logcpm_std.shape[1],
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
        wandb_run=wandb_run,
        batch_classes=batch_classes,
    )
    model = fit["model"]

    # Inference: corrected = decoder(encoder(X))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    with torch.no_grad():
        X_all = torch.tensor(logcpm_std.values, dtype=torch.float32).to(device)
        x_hat, _, _, z = model(X_all, adv_lambda=args.adv_weight)
        corr_std = x_hat.cpu().numpy()
        z_lat = z.cpu().numpy()

    # Inverse standardization -> logCPM scale
    corr_logcpm = inverse_standardize(corr_std, scaler)

    corrected_df = pd.DataFrame(corr_logcpm, index=logcpm_std.index, columns=logcpm_std.columns)
    corrected_df.to_csv(args.out_corrected)
    print(f"[OK] Wrote corrected matrix (logCPM) to: {args.out_corrected}")

    if args.out_latent:
        z_df = pd.DataFrame(z_lat, index=logcpm_std.index,
                            columns=[f"z{i+1}" for i in range(z_lat.shape[1])])
        z_df.to_csv(args.out_latent)
        print(f"[OK] Wrote latent embedding to: {args.out_latent}")

    if args.save_model:
        torch.save({"state_dict": model.state_dict(),
                    "batch_classes": batch_classes,
                    "label_classes": label_classes,
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

    # Quick diagnostics on latent space
    sil = quick_silhouettes(z_lat, batch_idx, label_idx)
    print(f"Silhouette (batch)  : {sil['sil_batch']:.3f}" if not np.isnan(sil['sil_batch']) else "Silhouette (batch): NA")
    print(f"Silhouette (label)  : {sil['sil_label']:.3f}" if not np.isnan(sil['sil_label']) else "Silhouette (label): NA")

    # Optionally generate PCA and boxplot visualisations using the visualiser module
    if args.generate_viz:
        try:
            # local import to avoid hard dependency
            from visualise import library_size_normalize as viz_libnorm, select_hvg as viz_select_hvg, plot_pca as viz_plot_pca, plot_boxplots as viz_plot_boxplots
        except Exception:
            try:
                # try package-style import if running as module
                from .visualise import library_size_normalize as viz_libnorm, select_hvg as viz_select_hvg, plot_pca as viz_plot_pca, plot_boxplots as viz_plot_boxplots
            except Exception as e:
                print(f"[VIZ] Could not import visualiser: {e}")
                return

        try:
            # compute before-correction logCPM from raw counts (same as earlier)
            logcpm_before = library_size_normalize(counts_raw)
            # select HVGs if requested for viz
            if args.viz_hvg_top and args.viz_hvg_top > 0:
                logcpm_before_sel = viz_select_hvg(logcpm_before, args.viz_hvg_top)
            else:
                logcpm_before_sel = logcpm_before

            # plot before
            viz_plot_pca(logcpm_before_sel, meta, args.batch_col, args.label_col if args.label_col in meta.columns else None, args.viz_pca_before, title=f'PCA (before) - HVG={args.viz_hvg_top}')

            # load corrected -> ensure samples x genes orientation
            corrected_read = pd.read_csv(args.out_corrected, index_col=0)
            if corrected_read.shape[0] != meta.shape[0] and corrected_read.shape[1] == meta.shape[0]:
                corrected_read = corrected_read.T

            # if HVG was used, restrict corrected to those genes when present
            if args.viz_hvg_top and args.viz_hvg_top > 0:
                genes = [g for g in logcpm_before_sel.columns if g in corrected_read.columns]
                if len(genes) >= 2:
                    corrected_for_pca = corrected_read.loc[:, genes]
                else:
                    corrected_for_pca = corrected_read
            else:
                corrected_for_pca = corrected_read

            viz_plot_pca(corrected_for_pca, meta, args.batch_col, args.label_col if args.label_col in meta.columns else None, args.viz_pca_after, title=f'PCA (after) - HVG={args.viz_hvg_top}')

            # align indices for boxplots
            both_idx = logcpm_before.index.intersection(corrected_read.index)
            viz_plot_boxplots(logcpm_before.loc[both_idx], corrected_read.loc[both_idx], meta.loc[both_idx], args.batch_col, args.viz_boxplot)
        except Exception as e:
            print(f"[VIZ] Failed to generate visualisations: {e}")


if __name__ == "__main__":
    main()
