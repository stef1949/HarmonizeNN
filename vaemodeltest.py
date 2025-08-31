#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Improved neural network-based batch correction for bulk RNA-seq.

This version incorporates a Variational Autoencoder (VAE) with a self-attention
mechanism and uses a Negative Binomial loss for reconstruction, which is
statistically more appropriate for count data.

Pipeline:
1) Read counts and metadata
2) Library-size normalize -> CPM -> log1p (for HVG selection)
3) Optional highly-variable gene (HVG) selection
4) Per-gene standardization (z-score) of raw counts for training stability
5) Train VAE with attention, gradient-reversal batch adversary,
   and a Negative Binomial reconstruction loss.
6) Export batch-corrected matrix (decoder's mean parameter) and latent space.
7) Perform SHAP analysis for interpretability.

Author: Steph Ritchie (Original), with modifications for VAE/Attention/NB/SHAP
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
from torch.distributions import NegativeBinomial
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, accuracy_score
from sklearn.model_selection import train_test_split

# SHAP is an external dependency: pip install shap
try:
    import shap
except ImportError:
    print("[Warning] SHAP library not found. Interpretability step will be skipped. Install with: pip install shap")
    shap = None


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


def standardize_per_gene(df: pd.DataFrame) -> Tuple[pd.DataFrame, StandardScaler]:
    """Standardize (z-score) a dataframe."""
    scaler = StandardScaler(with_mean=True, with_std=True)
    X = scaler.fit_transform(df.values)
    return pd.DataFrame(X, index=df.index, columns=df.columns), scaler


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
# --- IMPROVED VAE MODEL ---
# ----------------------------

def make_mlp(sizes, dropout=0.0, last_activation=None):
    layers = []
    for i in range(len(sizes) - 1):
        in_f, out_f = sizes[i], sizes[i+1]
        layers += [nn.Linear(in_f, out_f)]
        if i < len(sizes) - 2:  # hidden
            layers += [nn.ReLU(), nn.Dropout(dropout)]
        else: # last layer
            if last_activation == "relu":
                layers += [nn.ReLU()]
            elif last_activation == "tanh":
                layers += [nn.Tanh()]
            elif last_activation == "sigmoid":
                layers += [nn.Sigmoid()]
    return nn.Sequential(*layers)


class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads
        assert self.head_dim * heads == embed_size, "Embedding size must be divisible by heads"
        self.multihead_attn = nn.MultiheadAttention(embed_dim=embed_size, num_heads=heads, batch_first=True)
        self.layer_norm = nn.LayerNorm(embed_size)

    def forward(self, x):
        x_unsqueezed = x.unsqueeze(1)  # Shape: (batch, 1, features) for seq_len=1
        attention_output, _ = self.multihead_attn(x_unsqueezed, x_unsqueezed, x_unsqueezed)
        attention_output = self.layer_norm(attention_output + x_unsqueezed)
        return attention_output.squeeze(1)


class VaeAttentionBatchCorrector(nn.Module):
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
        attention_heads: int = 4,
    ):
        super().__init__()
        self.n_labels = n_labels
        self.grl = GradReverseLayer(lambda_=adv_lambda)

        # 1. Encoder with Self-Attention and VAE output
        enc_sizes = [n_genes] + list(enc_hidden)
        self.encoder_base = make_mlp(enc_sizes, dropout=dropout, last_activation="relu")
        self.attention = SelfAttention(enc_hidden[-1], attention_heads)
        self.fc_mu = nn.Linear(enc_hidden[-1], latent_dim)
        self.fc_log_var = nn.Linear(enc_hidden[-1], latent_dim)

        # 2. Decoder for Negative Binomial parameters
        dec_sizes_base = [latent_dim] + list(dec_hidden)
        self.decoder_base = make_mlp(dec_sizes_base, dropout=dropout, last_activation="relu")
        self.dec_mu = nn.Sequential(nn.Linear(dec_hidden[-1], n_genes), nn.Softplus())
        self.dec_theta = nn.Sequential(nn.Linear(dec_hidden[-1], n_genes), nn.Softplus())

        # 3. Adversary and Supervisor
        adv_sizes = [latent_dim] + list(adv_hidden) + [n_batches]
        self.adv = make_mlp(adv_sizes, dropout=dropout, last_activation=None)
        if n_labels is not None:
            sup_sizes = [latent_dim] + list(sup_hidden) + [n_labels]
            self.sup = make_mlp(sup_sizes, dropout=dropout, last_activation=None)
        else:
            self.sup = None

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, adv_lambda: Optional[float] = None):
        h = self.encoder_base(x)
        h_attn = self.attention(h)
        mu, log_var = self.fc_mu(h_attn), self.fc_log_var(h_attn)
        z = self.reparameterize(mu, log_var)
        
        h_dec = self.decoder_base(z)
        recon_mu = self.dec_mu(h_dec)
        recon_theta = self.dec_theta(h_dec)

        if adv_lambda is not None: self.grl.set_lambda(adv_lambda)
        z_rev = self.grl(z)
        batch_logits = self.adv(z_rev)
        label_logits = self.sup(z) if self.sup is not None else None

        return recon_mu, recon_theta, mu, log_var, batch_logits, label_logits, z


# ----------------------------
# --- UPDATED TRAINING & LOSS ---
# ----------------------------

def nb_loss(mu, theta, x):
    """Negative Log-Likelihood of the Negative Binomial distribution."""
    theta = torch.clamp(theta, min=1e-6, max=1e6) # Clamp for stability
    dist = NegativeBinomial(total_count=theta, logits=mu.log() - theta.log())
    return -dist.log_prob(x).sum(dim=-1).mean()

def kl_divergence_loss(mu, log_var):
    """KL divergence loss for VAE."""
    return -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=1).mean()


def train_model(
    model: VaeAttentionBatchCorrector,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 100,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    adv_weight: float = 1.0,
    sup_weight: float = 1.0,
    kl_weight: float = 0.001,
    adv_lambda_schedule: str = "linear",
    use_amp: bool = False,
    scheduler_type: str = "none",
    grad_accum_steps: int = 1,
    patience: int = 15,
    wandb_run: Optional[object] = None,
    save_best_path: Optional[Path] = None,
    batch_classes: Optional[list] = None,
) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    ce = nn.CrossEntropyLoss()
    scaler = GradScaler(enabled=use_amp and torch.cuda.is_available())

    if scheduler_type == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=5, min_lr=1e-6)
    elif scheduler_type == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    else:
        scheduler = None

    best_state, best_val, wait = None, float("inf"), 0
    history = {"train_loss": [], "val_loss": [], "val_batch_acc": [], "val_sup_acc": []}

    def lambda_at_epoch(t):
        if adv_lambda_schedule == "constant": return adv_weight
        return adv_weight * min(1.0, (t + 1) / max(1, epochs // 3))

    for epoch in range(epochs):
        model.train()
        lam = lambda_at_epoch(epoch)
        train_loss = 0.0
        opt.zero_grad()

        for step, batch in enumerate(train_loader):
            Xb, Bb, Lb = (batch[0], batch[1], batch[2] if len(batch) > 2 else None)
            Xb, Bb = Xb.to(device), Bb.to(device)
            Lb = Lb.to(device) if Lb is not None else None

            with torch.amp.autocast(device_type='cuda', enabled=use_amp and torch.cuda.is_available()):
                recon_mu, recon_theta, mu, log_var, b_logits, l_logits, _ = model(Xb, adv_lambda=lam)
                
                loss_recon = nb_loss(recon_mu, recon_theta, Xb)
                loss_kl = kl_divergence_loss(mu, log_var)
                loss_adv = ce(b_logits, Bb)
                loss = loss_recon + (kl_weight * loss_kl) + (adv_weight * loss_adv)
                
                if l_logits is not None and Lb is not None:
                    loss += sup_weight * ce(l_logits, Lb)

            effective_loss = loss / grad_accum_steps
            scaler.scale(effective_loss).backward()

            if ((step + 1) % grad_accum_steps == 0) or (step + 1 == len(train_loader)):
                scaler.step(opt)
                scaler.update()
                opt.zero_grad()
            train_loss += loss.item() * Xb.size(0)
        train_loss /= len(train_loader.dataset)

        # Validation
        model.eval()
        val_loss = 0.0
        all_b_true, all_b_pred, all_l_true, all_l_pred, all_z = [], [], [], [], []
        with torch.no_grad():
            for batch in val_loader:
                Xb, Bb, Lb = (batch[0], batch[1], batch[2] if len(batch) > 2 else None)
                Xb, Bb = Xb.to(device), Bb.to(device)
                Lb = Lb.to(device) if Lb is not None else None

                with torch.amp.autocast(device_type='cuda', enabled=use_amp and torch.cuda.is_available()):
                    recon_mu, recon_theta, mu, log_var, b_logits, l_logits, z = model(Xb, adv_lambda=lam)
                    loss_recon = nb_loss(recon_mu, recon_theta, Xb)
                    loss_kl = kl_divergence_loss(mu, log_var)
                    loss_adv = ce(b_logits, Bb)
                    loss = loss_recon + (kl_weight * loss_kl) + (adv_weight * loss_adv)
                    if l_logits is not None and Lb is not None:
                        loss += sup_weight * ce(l_logits, Lb)
                
                val_loss += loss.item() * Xb.size(0)
                all_b_true.append(Bb.cpu().numpy())
                all_b_pred.append(b_logits.argmax(dim=1).cpu().numpy())
                if l_logits is not None and Lb is not None:
                    all_l_true.append(Lb.cpu().numpy())
                    all_l_pred.append(l_logits.argmax(dim=1).cpu().numpy())
                all_z.append(z.cpu().numpy())

        val_loss /= len(val_loader.dataset)
        b_acc = accuracy_score(np.concatenate(all_b_true), np.concatenate(all_b_pred))
        l_acc = accuracy_score(np.concatenate(all_l_true), np.concatenate(all_l_pred)) if all_l_true else np.nan

        history.update({"train_loss": history.get("train_loss", []) + [train_loss],
                        "val_loss": history.get("val_loss", []) + [val_loss],
                        "val_batch_acc": history.get("val_batch_acc", []) + [b_acc],
                        "val_sup_acc": history.get("val_sup_acc", []) + [l_acc]})

        print(f"Epoch {epoch+1:03d}/{epochs} | train {train_loss:.4f} | val {val_loss:.4f} | "
              f"val batch acc {b_acc:.3f}" + (f" | val label acc {l_acc:.3f}" if not np.isnan(l_acc) else ""))

        if scheduler:
            scheduler.step(val_loss if scheduler_type == 'plateau' else None)

        if val_loss < best_val:
            best_val, wait = val_loss, 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            if save_best_path: torch.save({"state_dict": model.state_dict()}, save_best_path)
        else:
            wait += 1
            if wait >= patience:
                print("Early stopping.")
                break

    if best_state: model.load_state_dict(best_state)
    return {"model": model, "history": history}


# ----------------------------
# Evaluation and Main
# ----------------------------

def quick_silhouettes(z_latent: np.ndarray, batch_idx: np.ndarray, label_idx: Optional[np.ndarray]):
    out = {}
    try:
        out["sil_batch"] = silhouette_score(z_latent, batch_idx, metric="euclidean")
        if label_idx is not None and len(np.unique(label_idx)) > 1:
            out["sil_label"] = silhouette_score(z_latent, label_idx, metric="euclidean")
        else:
            out["sil_label"] = np.nan
    except Exception:
        out["sil_batch"] = out["sil_label"] = np.nan
    return out


def load_inputs(counts_path, meta_path, sample_col, batch_col, label_col, genes_in_rows):
    counts = pd.read_csv(counts_path, index_col=0)
    if genes_in_rows: counts = counts.T
    
    meta = pd.read_csv(meta_path).set_index(sample_col)
    common = counts.index.intersection(meta.index)
    if len(common) < 2:
        raise ValueError("Too few overlapping samples between counts and metadata.")
    
    return counts.loc[common], meta.loc[common]


def main():
    ap = argparse.ArgumentParser()
    # --- Input Args ---
    ap.add_argument("--counts", required=True, type=Path)
    ap.add_argument("--metadata", required=True, type=Path)
    ap.add_argument("--sample_col", default="sample", type=str)
    ap.add_argument("--batch_col", default="batch", type=str)
    ap.add_argument("--label_col", default=None, type=str)
    ap.add_argument("--genes_in_rows", action="store_true")
    ap.add_argument("--hvg", default=5000, type=int)
    # --- Model Args ---
    ap.add_argument("--latent_dim", default=32, type=int)
    ap.add_argument("--enc_hidden", default="1024,256", type=str)
    ap.add_argument("--dec_hidden", default="256,1024", type=str)
    ap.add_argument("--adv_hidden", default="128", type=str)
    ap.add_argument("--sup_hidden", default="64", type=str)
    ap.add_argument("--attention_heads", default=4, type=int)
    # --- Training Args ---
    ap.add_argument("--epochs", default=200, type=int)
    ap.add_argument("--lr", default=1e-3, type=float)
    ap.add_argument("--batch_size", default=64, type=int)
    ap.add_argument("--adv_weight", default=1.0, type=float)
    ap.add_argument("--sup_weight", default=1.0, type=float)
    ap.add_argument("--kl_weight", default=0.001, type=float)
    ap.add_argument("--weight_decay", default=0.0, type=float)
    ap.add_argument("--scheduler", default="plateau", choices=["none", "plateau", "cosine"])
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--dropout", default=0.1, type=float)
    ap.add_argument("--patience", default=20, type=int)
    ap.add_argument("--seed", default=42, type=int)
    # --- Output Args ---
    ap.add_argument("--out_corrected", required=True, type=Path)
    ap.add_argument("--out_latent", default=None, type=Path)
    ap.add_argument("--out_shap", default=None, type=Path)
    
    args = ap.parse_args()
    set_seed(args.seed)

    # 1. Load and Preprocess Data
    counts_raw, meta = load_inputs(args.counts, args.metadata, args.sample_col, args.batch_col, args.label_col, args.genes_in_rows)
    
    logcpm = library_size_normalize(counts_raw)
    hvg_genes = select_hvg(logcpm, args.hvg).columns
    counts_hvg = counts_raw[hvg_genes]
    
    # Standardize raw counts for model input
    counts_std, scaler = standardize_per_gene(counts_hvg)

    batch_cats = meta[args.batch_col].astype("category")
    batch_idx = batch_cats.cat.codes.values
    
    label_idx, label_classes = None, None
    if args.label_col:
        label_cats = meta[args.label_col].astype("category")
        label_idx = label_cats.cat.codes.values
        label_classes = label_cats.cat.categories.tolist()

    # 2. Create Datasets and Dataloaders
    train_ix, val_ix = train_test_split(np.arange(len(counts_std)), test_size=0.2, random_state=args.seed, stratify=batch_idx)
    
    ds_train = RNADataset(counts_std.values[train_ix], batch_idx[train_ix], label_idx[train_ix] if label_idx is not None else None)
    ds_val = RNADataset(counts_std.values[val_ix], batch_idx[val_ix], label_idx[val_ix] if label_idx is not None else None)

    dl_train = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True)
    dl_val = DataLoader(ds_val, batch_size=args.batch_size, shuffle=False)

    # 3. Initialize and Train Model
    enc_hidden = tuple(int(x) for x in args.enc_hidden.split(","))
    dec_hidden = tuple(int(x) for x in args.dec_hidden.split(","))
    adv_hidden = tuple(int(x) for x in args.adv_hidden.split(","))
    sup_hidden = tuple(int(x) for x in args.sup_hidden.split(","))

    model = VaeAttentionBatchCorrector(
        n_genes=counts_std.shape[1],
        latent_dim=args.latent_dim,
        enc_hidden=enc_hidden, dec_hidden=dec_hidden,
        adv_hidden=adv_hidden, sup_hidden=sup_hidden,
        n_batches=len(batch_cats.cat.categories),
        n_labels=len(label_classes) if label_classes else None,
        dropout=args.dropout,
        attention_heads=args.attention_heads
    )

    fit = train_model(
        model=model, train_loader=dl_train, val_loader=dl_val,
        epochs=args.epochs, lr=args.lr, weight_decay=args.weight_decay,
        adv_weight=args.adv_weight, sup_weight=args.sup_weight, kl_weight=args.kl_weight,
        use_amp=args.amp, scheduler_type=args.scheduler, patience=args.patience
    )
    model = fit["model"]

    # 4. Inference and Output
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    with torch.no_grad():
        X_all = torch.tensor(counts_std.values, dtype=torch.float32).to(device)
        recon_mu, _, _, _, _, _, z_lat = model(X_all)
        
        # The decoder's mean parameter `recon_mu` is our corrected counts
        corrected_counts = recon_mu.cpu().numpy()
        
    corrected_df = pd.DataFrame(corrected_counts, index=counts_std.index, columns=counts_std.columns)
    corrected_df.to_csv(args.out_corrected)
    print(f"[OK] Wrote corrected matrix (in count scale) to: {args.out_corrected}")

    if args.out_latent:
        z_df = pd.DataFrame(z_lat.cpu().numpy(), index=counts_std.index, columns=[f"z{i+1}" for i in range(z_lat.shape[1])])
        z_df.to_csv(args.out_latent)
        print(f"[OK] Wrote latent embedding to: {args.out_latent}")

    # 5. Diagnostics and SHAP Interpretability
    sil = quick_silhouettes(z_lat.cpu().numpy(), batch_idx, label_idx)
    print(f"Silhouette (batch): {sil['sil_batch']:.3f}")
    print(f"Silhouette (label): {sil['sil_label']:.3f}")

    if args.out_shap and shap is not None:
        print("\n[INFO] Starting SHAP analysis...")
        background = torch.tensor(counts_std.values[train_ix[:100]], dtype=torch.float32).to(device) # background from 100 train samples
        test_samples = torch.tensor(counts_std.values[val_ix[:20]], dtype=torch.float32).to(device) # explain 20 val samples

        def adv_predictor(x):
            _, _, _, _, b_logits, _, _ = model(x)
            return b_logits
        
        explainer = shap.DeepExplainer(adv_predictor, background)
        shap_values = explainer.shap_values(test_samples)

        mean_abs_shap = np.abs(shap_values).mean(axis=(0, 1))
        shap_df = pd.DataFrame({'gene': counts_std.columns, 'shap_importance': mean_abs_shap})
        shap_df = shap_df.sort_values(by='shap_importance', ascending=False)

        print("\n--- Top 10 Genes Driving Batch Effects (SHAP) ---")
        print(shap_df.head(10))
        shap_df.to_csv(args.out_shap, index=False)
        print(f"[OK] Saved full SHAP importance to: {args.out_shap}")


if __name__ == "__main__":
    main()