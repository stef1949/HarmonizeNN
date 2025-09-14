#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Improved neural network-based batch correction for bulk RNA-seq with W&B support.

This version incorporates a Variational Autoencoder (VAE) with a self-attention
mechanism, uses a Negative Binomial loss, and integrates with Weights & Biases
for comprehensive experiment tracking and visualization.

Author: Steph Ritchie (Original), with modifications for VAE/Attention/NB/SHAP/W&B
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
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import NearestNeighbors

# Optional dependencies: pip install shap wandb
try:
    import shap
except ImportError:
    print("[Warning] SHAP library not found. Interpretability step will be skipped. Install with: pip install shap")
    shap = None
try:
    import wandb
except ImportError:
    print("[Warning] W&B library not found. Tracking will be disabled. Install with: pip install wandb")
    wandb = None


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
    lib_sizes = counts_df.sum(axis=1).replace(0, np.nan)
    x = counts_df.div(lib_sizes, axis=0) * cpm_factor
    return np.log1p(x)


def select_hvg(logcpm_df: pd.DataFrame, n_hvg: int) -> pd.DataFrame:
    if n_hvg <= 0 or n_hvg >= logcpm_df.shape[1]:
        return logcpm_df
    vars_ = logcpm_df.var(axis=0)
    top = vars_.nlargest(n_hvg).index
    return logcpm_df.loc[:, top]


def standardize_per_gene(df: pd.DataFrame) -> Tuple[pd.DataFrame, StandardScaler]:
    scaler = StandardScaler(with_mean=True, with_std=True)
    X = scaler.fit_transform(df.values)
    return pd.DataFrame(X, index=df.index, columns=df.columns), scaler

def safe_silhouette(X, labels):
    # Need >=2 distinct labels and no label with all samples
    if X.shape[0] < 3:
        return 0.0
    uniq = np.unique(labels)
    if len(uniq) < 2:
        return 0.0
    # Require at least one sample in two clusters
    try:
        return silhouette_score(X, labels, metric="euclidean")
    except Exception:
        return 0.0

# (Silhouette metrics are computed after each validation epoch once latent vectors are collected.)

def bio_center_loss(z: torch.Tensor, labels: Optional[torch.Tensor], gamma: float = 0.5) -> torch.Tensor:
    if labels is None or z is None or z.numel() == 0:
        return z.new_tensor(0.0)
    classes = torch.unique(labels)
    if classes.numel() < 2:
        return z.new_tensor(0.0)
    within = z.new_tensor(0.0)
    centers = []
    for c in classes:
        mask = (labels == c)
        if mask.sum() == 0:
            continue
        zc = z[mask]
        center = zc.mean(dim=0, keepdim=True)
        centers.append(center)
        diff = zc - center
        within = within + (diff.pow(2).sum(dim=1).mean())
    if len(centers) == 0:
        return z.new_tensor(0.0)
    within = within / max(1, len(centers))
    centers_t = torch.cat(centers, dim=0)
    if centers_t.size(0) > 1:
        diff = centers_t.unsqueeze(1) - centers_t.unsqueeze(0)
        d2 = diff.pow(2).sum(dim=2)
        C = centers_t.size(0)
        between = (d2.sum() - d2.diag().sum()) / max(1, C * (C - 1))
    else:
        between = z.new_tensor(0.0)
    return within - gamma * between
def _knn_stats(Z: np.ndarray, labels: np.ndarray, k: int = 15):
    """Compute kNN entropy (normalized), LISI, and kNN majority accuracy.
    Works with string labels by mapping to integer codes first.
    """
    try:
        n = Z.shape[0]
        if n < 3 or labels is None:
            return np.nan, np.nan, np.nan
        # Map arbitrary labels -> integer codes 0..C-1
        labels = np.asarray(labels)
        classes, inv = np.unique(labels, return_inverse=True)
        C = int(len(classes))
        if C <= 1:
            return np.nan, np.nan, np.nan
        k_eff = min(int(k), max(2, n - 1))
        nn = NearestNeighbors(n_neighbors=k_eff + 1, algorithm="auto").fit(Z)
        idx = nn.kneighbors(Z, return_distance=False)[:, 1:]
        neigh_codes = inv[idx]  # shape (n, k)
        H = []  # entropy
        L = []  # LISI
        correct = 0
        for i in range(n):
            counts = np.bincount(neigh_codes[i], minlength=C).astype(float)
            p = counts / max(1.0, counts.sum())
            lisi = 1.0 / np.sum(p * p) if counts.sum() > 0 else np.nan
            L.append(lisi)
            mask = p > 0
            h = -np.sum(p[mask] * np.log(p[mask] + 1e-12)) if counts.sum() > 0 else np.nan
            h_norm = h / np.log(C) if (C > 1 and not np.isnan(h)) else np.nan
            H.append(h_norm)
            maj = int(np.argmax(counts)) if counts.sum() > 0 else -1
            correct += (maj == int(inv[i]))
        return float(np.nanmean(H)), float(np.nanmean(L)), float(correct / n)
    except Exception:
        return np.nan, np.nan, np.nan

def _eta2_by_group(Z: np.ndarray, groups: np.ndarray):
    try:
        n, d = Z.shape
        if n < 3:
            return np.nan
        overall_mean = Z.mean(axis=0)
        var_total = Z.var(axis=0, ddof=1) + 1e-12
        eta2_dims = []
        for g in np.unique(groups):
            pass
        # Between-group sum of squares per dim
        ssb = np.zeros(d, dtype=float)
        for g in np.unique(groups):
            idx = (groups == g)
            ng = idx.sum()
            if ng == 0:
                continue
            diff = Z[idx].mean(axis=0) - overall_mean
            ssb += ng * (diff ** 2)
        eta2 = (ssb / (n * var_total)).mean()
        return float(eta2)
    except Exception:
        return np.nan

# ----------------------------
# Data & Model Components
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


def make_mlp(sizes, dropout=0.0, last_activation=None):
    layers = []
    for i in range(len(sizes) - 1):
        in_f, out_f = sizes[i], sizes[i+1]
        layers += [nn.Linear(in_f, out_f)]
        if i < len(sizes) - 2:
            layers += [nn.ReLU(), nn.Dropout(dropout)]
        elif last_activation:
            layers += [getattr(nn, last_activation)()]
    return nn.Sequential(*layers)


class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(SelfAttention, self).__init__()
        self.multihead_attn = nn.MultiheadAttention(embed_dim=embed_size, num_heads=heads, batch_first=True)
        self.layer_norm = nn.LayerNorm(embed_size)

    def forward(self, x):
        x_unsqueezed = x.unsqueeze(1)
        attention_output, _ = self.multihead_attn(x_unsqueezed, x_unsqueezed, x_unsqueezed)
        attention_output = self.layer_norm(attention_output + x_unsqueezed)
        return attention_output.squeeze(1)


class VaeAttentionBatchCorrector(nn.Module):
    def __init__(self, n_genes: int, latent_dim: int, enc_hidden: tuple, dec_hidden: tuple,
                 adv_hidden: tuple, sup_hidden: tuple, n_batches: int, n_labels: Optional[int],
                 dropout: float, attention_heads: int):
        super().__init__()
        self.grl = GradReverseLayer(lambda_=1.0)

        # Encoder
        enc_sizes = [n_genes] + list(enc_hidden)
        self.encoder_base = make_mlp(enc_sizes, dropout=dropout, last_activation="ReLU")
        self.attention = SelfAttention(enc_hidden[-1], attention_heads)
        self.fc_mu = nn.Linear(enc_hidden[-1], latent_dim)
        self.fc_log_var = nn.Linear(enc_hidden[-1], latent_dim)

        # Decoder
        dec_sizes = [latent_dim] + list(dec_hidden)
        self.decoder_base = make_mlp(dec_sizes, dropout=dropout, last_activation="ReLU")
        self.dec_mu = nn.Sequential(nn.Linear(dec_hidden[-1], n_genes), nn.Softplus())
        self.dec_theta = nn.Sequential(nn.Linear(dec_hidden[-1], n_genes), nn.Softplus())

        # Adversary and Supervisor
        self.adv = make_mlp([latent_dim] + list(adv_hidden) + [n_batches], dropout)
        self.sup = make_mlp([latent_dim] + list(sup_hidden) + [n_labels], dropout) if n_labels else None

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, adv_lambda: Optional[float] = None):
        h = self.encoder_base(x)
        h_attn = self.attention(h)
        mu, log_var = self.fc_mu(h_attn), self.fc_log_var(h_attn)
        z = self.reparameterize(mu, log_var)
        
        recon_mu = self.dec_mu(self.decoder_base(z))
        recon_theta = self.dec_theta(self.decoder_base(z))

        if adv_lambda is not None: self.grl.set_lambda(adv_lambda)
        z_rev = self.grl(z)
        batch_logits = self.adv(z_rev)
        label_logits = self.sup(z) if self.sup else None

        return recon_mu, recon_theta, mu, log_var, batch_logits, label_logits, z


# ----------------------------
# Training Loop
# ----------------------------

def nb_loss(mu, theta, x, eps: float = 1e-8):
    """Vectorised NB log-likelihood (faster than torch.distributions for large tensors).
    mu, theta, x >=0; returns mean NLL over batch.
    """
    mu = mu.clamp_min(1e-5)
    theta = theta.clamp_min(1e-4)
    # log( (theta/(theta+mu))^theta * (mu/(theta+mu))^x * C(x+theta-1, x) )
    log_theta_mu = torch.log(theta + mu + eps)
    res = (
        theta * (torch.log(theta + eps) - log_theta_mu)
        + x * (torch.log(mu + eps) - log_theta_mu)
        + torch.lgamma(x + theta)
        - torch.lgamma(theta)
        - torch.lgamma(x + 1.0)
    )
    return -res.sum(-1).mean()


def kl_divergence_loss(mu, log_var):
    return -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=1).mean()


def train_model(model: VaeAttentionBatchCorrector, train_loader: DataLoader, val_loader: DataLoader,
                epochs: int, lr: float, weight_decay: float, adv_weight: float,
                sup_weight: float, kl_weight: float, use_amp: bool, scheduler_type: str,
                patience: int, device: torch.device, save_best_path: Optional[Path] = None,
                wandb_run: Optional[object] = None, batch_classes: Optional[list] = None,
                log_latent_every: int = 0, grad_accum_steps: int = 1, warmup_ratio: float = 0.1,
                log_lr_steps: bool = False, use_fused_optim: bool = False, use_cuda_graphs: bool = False,
                precision_dtype: torch.dtype = torch.float32,
                cond_min: float = 0.10, overcorr_weight: float = 0.5,
                adaptive_cond_margin: float = 0.02, adaptive_cond_scale: float = 0.85,
                bio_weight: float = 0.0, bio_gamma: float = 0.5):
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    # Optimizer (optionally fused AdamW)
    if use_fused_optim:
        try:
            opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, fused=True)
            print("[INFO] Using fused AdamW optimizer")
        except Exception as e:
            print(f"[WARN] Fused AdamW unavailable ({e}); falling back to standard AdamW")
            opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    ce = nn.CrossEntropyLoss()
    # Only enable AMP scaler if CUDA is available; avoids noisy warnings on CPU-only systems
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp and torch.cuda.is_available())

    # CUDA Graphs constraints / preparation
    if use_cuda_graphs and not torch.cuda.is_available():
        print("[WARN] --cuda_graphs ignored (no CUDA device)")
        use_cuda_graphs = False
    if use_cuda_graphs and grad_accum_steps > 1:
        print("[WARN] Disabling grad accumulation because CUDA Graphs is enabled")
        grad_accum_steps = 1
    graph = None
    static_X = static_B = static_L = None
    capture_done = False
    # Scheduler setup (adds cosine_warmup)
    if scheduler_type == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=5)
    elif scheduler_type == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    elif scheduler_type == "cosine_warmup":
        import math
        steps_per_epoch = (len(train_loader) + grad_accum_steps - 1) // grad_accum_steps
        total_steps = max(1, epochs * steps_per_epoch)
        warmup_steps = max(1, int(total_steps * warmup_ratio))
        def lr_lambda(step: int):
            if step < warmup_steps:
                return float(step) / float(warmup_steps)
            progress = (step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            return 0.5 * (1.0 + math.cos(math.pi * progress))
        scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lr_lambda)
    else:
        scheduler = None

    best_val_loss, wait = float("inf"), 0

    import time
    optimizer_steps = 0
    last_epoch_cond_sil: Optional[float] = None
    for epoch in range(epochs):
        t0 = time.time()
        model.train()
        lam = adv_weight * min(1.0, (epoch + 1) / max(1, epochs // 3))
        try:
            if last_epoch_cond_sil is not None and (last_epoch_cond_sil < (cond_min + adaptive_cond_margin)):
                lam = max(adv_weight * 0.05, lam * adaptive_cond_scale)
        except Exception:
            pass
        opt.zero_grad(set_to_none=True)
        # ---- Batch loop ----
        for batch_idx, batch in enumerate(train_loader):
            if not use_cuda_graphs:
                Xb, Bb, Lb = (batch[0].to(device), batch[1].to(device), batch[2].to(device) if len(batch) > 2 else None)
                with torch.amp.autocast(device_type='cuda', dtype=precision_dtype, enabled=use_amp and torch.cuda.is_available()):
                    recon_mu, recon_theta, mu, log_var, b_logits, l_logits, z = model(Xb, adv_lambda=lam)
                    loss_r = nb_loss(recon_mu, recon_theta, Xb)
                    loss_kl = kl_divergence_loss(mu, log_var)
                    loss_a = ce(b_logits, Bb)
                    total_loss = loss_r + (kl_weight * loss_kl) + (adv_weight * loss_a)
                    if l_logits is not None:
                        total_loss += sup_weight * ce(l_logits, Lb)
                    if (bio_weight > 0.0) and (Lb is not None):
                        total_loss += float(bio_weight) * bio_center_loss(z, Lb, gamma=bio_gamma)
                    loss = total_loss
                if grad_accum_steps > 1:
                    loss = loss / grad_accum_steps
                scaler.scale(loss).backward()
                do_step = ((batch_idx + 1) % grad_accum_steps == 0) or (batch_idx + 1 == len(train_loader))
                if do_step:
                    scaler.step(opt)
                    scaler.update()
                    opt.zero_grad(set_to_none=True)
                    optimizer_steps += 1
                    if scheduler and scheduler_type == 'cosine_warmup':
                        scheduler.step()
                    if log_lr_steps or (wandb_run and getattr(wandb_run, 'config', None) and getattr(wandb_run.config, 'log_lr_steps', False)):
                        lr_now = opt.param_groups[0]['lr']
                        try:
                            if wandb_run:
                                wandb_run.log({"lr_step": lr_now, "train_loss_step": float(total_loss.detach())}, step=optimizer_steps)
                            else:
                                print(f"Step {optimizer_steps:06d} LR {lr_now:.2e} Loss {float(total_loss.detach()):.4f}")
                        except Exception:
                            pass
                    elif not wandb_run and optimizer_steps % 200 == 0:
                        print(f"Step {optimizer_steps:06d} Loss {float(total_loss.detach()):.4f}")
            else:
                # CUDA Graphs path
                if not capture_done:
                    Xb, Bb, Lb = (batch[0].to(device), batch[1].to(device), batch[2].to(device) if len(batch) > 2 else None)
                    # Warmup (optimizer state init)
                    opt.zero_grad(set_to_none=True)
                    with torch.amp.autocast(device_type='cuda', dtype=precision_dtype, enabled=use_amp and torch.cuda.is_available()):
                        recon_mu, recon_theta, mu, log_var, b_logits, l_logits, _ = model(Xb, adv_lambda=lam)
                        warm_loss = nb_loss(recon_mu, recon_theta, Xb) + kl_weight * kl_divergence_loss(mu, log_var) + adv_weight * ce(b_logits, Bb)
                        if l_logits is not None:
                            warm_loss = warm_loss + sup_weight * ce(l_logits, Lb)
                    warm_loss.backward(); opt.step(); opt.zero_grad(set_to_none=True)
                    torch.cuda.synchronize()
                    static_X = Xb.clone(); static_B = Bb.clone(); static_L = Lb.clone() if Lb is not None else None
                    graph = torch.cuda.CUDAGraph()
                    # Capture
                    with torch.cuda.graph(graph):
                        recon_mu_g, recon_theta_g, mu_g, log_var_g, b_logits_g, l_logits_g, _ = model(static_X, adv_lambda=lam)
                        g_loss = nb_loss(recon_mu_g, recon_theta_g, static_X) + kl_weight * kl_divergence_loss(mu_g, log_var_g) + adv_weight * ce(b_logits_g, static_B)
                        if static_L is not None and l_logits_g is not None:
                            g_loss = g_loss + sup_weight * ce(l_logits_g, static_L)
                        g_loss.backward(); opt.step(); opt.zero_grad(set_to_none=True)
                    capture_done = True
                    optimizer_steps += 1
                else:
                    Xb, Bb, Lb = (batch[0].to(device), batch[1].to(device), batch[2].to(device) if len(batch) > 2 else None)
                    if Xb.shape == static_X.shape:
                        static_X.copy_(Xb); static_B.copy_(Bb)
                        if static_L is not None and Lb is not None and static_L.shape == Lb.shape:
                            static_L.copy_(Lb)
                        graph.replay(); optimizer_steps += 1
                    else:
                        print("[WARN] Batch shape changed; skipping CUDA Graph replay this batch")

        # Validation
        model.eval()
        val_loss, val_r, val_kl, val_a, val_s = 0.0, 0.0, 0.0, 0.0, 0.0
        all_b_true, all_b_pred, all_l_true, all_l_pred, all_z = [], [], [], [], []
        with torch.no_grad():
            for batch in val_loader:
                Xb, Bb, Lb = (batch[0].to(device), batch[1].to(device), batch[2].to(device) if len(batch) > 2 else None)
                recon_mu, recon_theta, mu, log_var, b_logits, l_logits, z = model(Xb)
                
                loss_r = nb_loss(recon_mu, recon_theta, Xb)
                loss_kl = kl_divergence_loss(mu, log_var)
                loss_a = ce(b_logits, Bb)
                loss = loss_r + (kl_weight * loss_kl) + (adv_weight * loss_a)
                val_r += loss_r.item(); val_kl += loss_kl.item(); val_a += loss_a.item()
                if l_logits is not None:
                    loss_s = ce(l_logits, Lb)
                    loss += sup_weight * loss_s
                    val_s += loss_s.item()
                
                val_loss += loss.item()
                all_b_true.extend(Bb.cpu().numpy()); all_b_pred.extend(b_logits.argmax(dim=1).cpu().numpy())
                if l_logits is not None:
                    all_l_true.extend(Lb.cpu().numpy()); all_l_pred.extend(l_logits.argmax(dim=1).cpu().numpy())
                all_z.append(z.cpu().numpy())
        
        val_loss /= len(val_loader)
        b_acc = accuracy_score(all_b_true, all_b_pred)
        l_acc = accuracy_score(all_l_true, all_l_pred) if all_l_true else np.nan

        # Compute silhouette metrics on latent space (only after validation collections)
        try:
            all_z_np = np.concatenate(all_z, axis=0)
            all_b_np = np.array(all_b_true)
            if all_z_np.shape[0] > 2:
                batch_sil = safe_silhouette(all_z_np, all_b_np)
                if all_l_true:
                    all_l_np = np.array(all_l_true)
                    cond_sil = safe_silhouette(all_z_np, all_l_np)
                    knn_lab_ent, clisi, knn_lab_acc = _knn_stats(all_z_np, all_l_np, k=15)
                    label_eta2 = _eta2_by_group(all_z_np, all_l_np)
                else:
                    all_l_np = None
                    cond_sil = np.nan
                    knn_lab_ent = np.nan
                    clisi = np.nan
                    knn_lab_acc = np.nan
                    label_eta2 = np.nan
                cond_minus_batch = (0.0 if np.isnan(cond_sil) else cond_sil) - batch_sil
                knn_batch_ent, ilisi, _ = _knn_stats(all_z_np, all_b_np, k=15)
                batch_eta2 = _eta2_by_group(all_z_np, all_b_np)
            else:
                batch_sil = 0.0
                cond_sil = np.nan
                cond_minus_batch = 0.0
                knn_batch_ent = ilisi = batch_eta2 = np.nan
                knn_lab_ent = clisi = knn_lab_acc = label_eta2 = np.nan
        except Exception:
            batch_sil = 0.0
            cond_sil = np.nan
            cond_minus_batch = 0.0
            knn_batch_ent = ilisi = batch_eta2 = np.nan
            knn_lab_ent = clisi = knn_lab_acc = label_eta2 = np.nan

        # Penalised objective to preserve biology (discourage low condition silhouette)
        try:
            cond_val = 0.0 if np.isnan(cond_sil) else float(cond_sil)
        except Exception:
            cond_val = 0.0
        objective_overcorr = cond_minus_batch - float(args.overcorr_weight) * max(0.0, float(args.cond_min) - cond_val)

        try:
            last_epoch_cond_sil = 0.0 if np.isnan(cond_sil) else float(cond_sil)
        except Exception:
            last_epoch_cond_sil = None
        t1 = time.time()
        current_lr = opt.param_groups[0]['lr']
        print(f"E {epoch+1:03d} | VLoss {val_loss:.4f} | BAcc {b_acc:.3f}" + (f" | LAcc {l_acc:.3f}" if not np.isnan(l_acc) else "") + f" | LR {current_lr:.2e} | Time {t1-t0:.2f}s")

        if wandb_run:
            log_dict = {"epoch": epoch, "val_loss": val_loss, "val_batch_acc": b_acc,
                        "val_recon_loss": val_r/len(val_loader), "val_kl_loss": val_kl/len(val_loader),
                        "val_adv_loss": val_a/len(val_loader), "grl_lambda": lam,
                        "val/batch_sil": batch_sil,
                        "val/cond_sil": 0.0 if np.isnan(cond_sil) else cond_sil,
                        "val/cond_minus_batch": cond_minus_batch,
                        "val/knn_batch_entropy": float(np.nan_to_num(knn_batch_ent, nan=0.0)),
                        "val/ilisi_batch": float(np.nan_to_num(ilisi, nan=0.0)),
                        "val/batch_eta2": float(np.nan_to_num(batch_eta2, nan=0.0)),
                        "objective_overcorr": objective_overcorr,
                       }
            if not np.isnan(l_acc):
                log_dict["val_label_acc"] = l_acc
                log_dict["label/knn_entropy"] = float(np.nan_to_num(knn_lab_ent, nan=0.0))
                log_dict["label/knn_acc"] = float(np.nan_to_num(knn_lab_acc, nan=0.0))
                log_dict["label/clisi"] = float(np.nan_to_num(clisi, nan=0.0))
                log_dict["label/eta2"] = float(np.nan_to_num(label_eta2, nan=0.0))
            if log_latent_every > 0 and ((epoch + 1) % log_latent_every == 0):
                try:
                    Z_cat = np.concatenate(all_z, axis=0)
                    pca_z = PCA(n_components=2).fit_transform(Z_cat)
                    fig, ax = plt.subplots(figsize=(5, 4))
                    for i, label in enumerate(batch_classes):
                        mask = np.array(all_b_true) == i
                        ax.scatter(pca_z[mask, 0], pca_z[mask, 1], label=label, alpha=0.75, s=12)
                    ax.legend(fontsize='x-small'); ax.set_title(f'Latent PCA E{epoch+1}'); ax.set_xticks([]); ax.set_yticks([])
                    log_dict["latent_space_pca"] = wandb.Image(fig)
                    plt.close(fig)
                except Exception as e:
                    print(f"[WARN] Latent viz failed: {e}")
            wandb_run.log(log_dict)
            wandb_run.log({
                "val/cond_sil": cond_sil,
                "val/batch_sil": batch_sil,
                "val/cond_minus_batch": cond_minus_batch,
                "objective_overcorr": objective_overcorr,
                "epoch": epoch + 1,
                "val/loss": val_loss,
            })

        if val_loss < best_val_loss:
            best_val_loss, wait = val_loss, 0
            if save_best_path:
                torch.save(model.state_dict(), save_best_path)
        else:
            wait += 1
            if wait >= patience:
                print("Early stopping.")
                break

        if scheduler and optimizer_steps > 0 and scheduler_type in ('plateau', 'cosine'):
            if scheduler_type == 'plateau':
                scheduler.step(val_loss)
            elif scheduler_type == 'cosine':
                scheduler.step()

    if save_best_path and os.path.exists(save_best_path): model.load_state_dict(torch.load(save_best_path))
    return {"model": model}


# ----------------------------
# Main Execution
# ----------------------------
def _train_val_split_indices(n: int, test_size: float, seed: int,
                             label_idx: Optional[np.ndarray], batch_idx: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return train/val indices ensuring val has both condition classes when available.
    Tries stratifying by (batch,label) pairs, then by label, then by batch, with fallbacks.
    """
    rng = np.random.RandomState(seed)
    val_size = max(1, int(round(test_size * n)))
    if label_idx is not None and len(np.unique(label_idx)) < 2:
        label_idx = None  # no stratification possible on labels
    # Candidate stratification targets
    y_pair = None
    if (label_idx is not None) and (batch_idx is not None) and (len(np.unique(batch_idx)) >= 2):
        n_labels = int(len(np.unique(label_idx)))
        y_pair = (batch_idx.astype(int) * n_labels) + label_idx.astype(int)
    for y in (y_pair, label_idx, batch_idx, None):
        try:
            if y is not None:
                splitter = StratifiedShuffleSplit(n_splits=1, test_size=val_size, random_state=seed)
                tr, va = next(splitter.split(np.zeros(n), y))
            else:
                idx = rng.permutation(n)
                va, tr = idx[:val_size], idx[val_size:]
            # Validate label coverage if labels available
            if label_idx is None:
                return tr, va
            if len(np.unique(label_idx[va])) >= 2 and len(np.unique(label_idx[tr])) >= 2:
                return tr, va
        except Exception:
            continue
    # Last-resort: Force at least one of each label in val
    if label_idx is not None:
        classes = np.unique(label_idx)
        chosen = []
        for c in classes:
            idx_c = np.where(label_idx == c)[0]
            if len(idx_c) > 0:
                chosen.append(rng.choice(idx_c))
        # Fill remainder randomly
        remaining = [i for i in range(n) if i not in set(chosen)]
        rng.shuffle(remaining)
        need = max(0, val_size - len(chosen))
        va = np.array(chosen + remaining[:need], dtype=int)
        tr = np.array([i for i in range(n) if i not in set(va)], dtype=int)
        return tr, va
    # If no labels, return random split
    idx = rng.permutation(n)
    va, tr = idx[:val_size], idx[val_size:]
    return tr, va
def load_inputs(counts_path, meta_path, sample_col, batch_col, label_col, genes_in_rows):
    """Loads, validates, and aligns counts and metadata."""
    counts = pd.read_csv(counts_path, index_col=0)
    if genes_in_rows:
        counts = counts.T
    
    meta = pd.read_csv(meta_path)
    if sample_col not in meta.columns:
        raise ValueError(f"Sample column '{sample_col}' not found in metadata.")
    meta = meta.set_index(sample_col)

    # Find common samples between the two files
    common_samples = counts.index.intersection(meta.index)
    
    if len(common_samples) < 2:
        raise ValueError("Found fewer than 2 overlapping samples in counts and metadata files.")
    
    # Return the aligned dataframes
    return counts.loc[common_samples], meta.loc[common_samples]

# ----------------------------
# Visualisation Helpers
# ----------------------------

def _pca_plot(df: pd.DataFrame, meta: pd.DataFrame, batch_col: str, label_col: Optional[str], out_path: Path, title: str):
    try:
        pca = PCA(n_components=2)
        Z = pca.fit_transform(df.values)
        pc_df = pd.DataFrame(Z, index=df.index, columns=["PC1", "PC2"])
        batches = meta.loc[pc_df.index, batch_col].astype("category")
        batch_codes = batches.cat.codes.values
        batch_names = batches.cat.categories.tolist()
        import matplotlib.pyplot as plt  # local import safe
        fig, ax = plt.subplots(figsize=(6,5))
        cmap = plt.get_cmap('tab10')
        for c in np.unique(batch_codes):
            mask = batch_codes == c
            lbl = batch_names[int(c)] if int(c) < len(batch_names) else str(int(c))
            ax.scatter(pc_df.loc[mask, 'PC1'], pc_df.loc[mask, 'PC2'], s=18, color=cmap(int(c)%cmap.N), label=lbl, alpha=0.85)
        if label_col and label_col in meta.columns:
            labels = meta.loc[pc_df.index, label_col].astype('category')
            if len(labels.cat.categories) > 1:
                # If classic tumor/normal (case-insensitive), map to fixed shapes; else cycle
                cats = list(labels.cat.categories)
                lower = [c.lower() for c in cats]
                fixed_map = {}
                if 'tumor' in lower and 'normal' in lower:
                    for c in cats:
                        if c.lower() == 'tumor':
                            fixed_map[c] = '^'  # triangle
                        elif c.lower() == 'normal':
                            fixed_map[c] = 'o'  # circle
                markers_cycle = ["o","s","^","D","v","P","X","*","<",">"]
                for i, cat in enumerate(cats):
                    m = fixed_map.get(cat, markers_cycle[i % len(markers_cycle)])
                    mask = labels == cat
                    ax.scatter(
                        pc_df.loc[mask, 'PC1'], pc_df.loc[mask, 'PC2'],
                        facecolors='none', edgecolors='k', marker=m, s=70, linewidths=0.9,
                        label=f"{cat}" if cat not in batch_names else f"{cat} (label)"
                    )
        ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
        ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
        ax.set_title(title)
        ax.legend(title='batch', bbox_to_anchor=(1.05,1), loc='upper left', fontsize='small')
        fig.tight_layout()
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"[OK] Saved PCA plot: {out_path}")
    except Exception as e:
        print(f"[VIZ] PCA failed ({out_path}): {e}")


def _boxplots(logcpm_before: pd.DataFrame, logcpm_after: pd.DataFrame, meta: pd.DataFrame, batch_col: str, out_path: Path):
    try:
        batches = meta[batch_col].astype('category')
        batch_names = batches.cat.categories.tolist()
        before_groups = []
        after_groups = []
        labels = []
        for b in batch_names:
            samp_ids = meta.index[meta[batch_col] == b].tolist()
            vals_before = logcpm_before.loc[samp_ids].values.flatten()
            vals_after = logcpm_after.loc[samp_ids].values.flatten()
            before_groups.append(vals_before)
            after_groups.append(vals_after)
            labels.append(b)
        import matplotlib.pyplot as plt
        from matplotlib import gridspec
        n = len(batch_names)
        fig = plt.figure(figsize=(max(6, n*1.3),5))
        gs = gridspec.GridSpec(1,1)
        ax = fig.add_subplot(gs[0])
        positions = []
        data = []
        tick_pos = []
        width = 0.35
        for i in range(n):
            base = i*2
            positions.extend([base, base+width])
            data.extend([before_groups[i], after_groups[i]])
            tick_pos.append(base+width/2)
        bplots = ax.boxplot(data, positions=positions, widths=width, patch_artist=True, showfliers=False)
        for idx, patch in enumerate(bplots['boxes']):
            patch.set_facecolor('#a6cee3' if idx%2==0 else '#b2df8a')
        ax.set_xticks(tick_pos)
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.set_ylabel('logCPM')
        ax.set_title('logCPM distributions by batch (before=blue, after=green)')
        fig.tight_layout()
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"[OK] Saved boxplots: {out_path}")
    except Exception as e:
        print(f"[VIZ] Boxplots failed ({out_path}): {e}")

def _pca_panel(before_df: pd.DataFrame, after_df: pd.DataFrame, meta: pd.DataFrame, batch_col: str, label_col: Optional[str], out_path: Path, hvg_desc: str):
    try:
        from sklearn.decomposition import PCA
        import matplotlib.pyplot as plt
        # Align samples across before/after and meta
        common = before_df.index.intersection(after_df.index)
        if len(common) < 2:
            raise ValueError("PCA panel requires >=2 overlapping samples between before and after")
        Bf = before_df.loc[common]
        Af = after_df.loc[common]
        # Align gene/features by intersection to avoid shape mismatch
        common_genes = Bf.columns.intersection(Af.columns)
        if len(common_genes) < 2:
            raise ValueError("PCA panel requires >=2 overlapping genes between before and after matrices")
        Bf = Bf.loc[:, common_genes]
        Af = Af.loc[:, common_genes]
        batches = meta.loc[common, batch_col].astype('category')
        # Fit a single PCA on concatenated data so axes are comparable
        from numpy import vstack
        pca = PCA(n_components=2).fit(vstack([Bf.values, Af.values]))
        Zb = pca.transform(Bf.values)
        Za = pca.transform(Af.values)
        batch_codes = batches.cat.codes.values
        batch_names = batches.cat.categories.tolist()
        cmap = plt.get_cmap('tab10')
        fig, axes = plt.subplots(1,2, figsize=(10,5))
        # shared axis limits for comparability
        allZ = np.vstack([Zb, Za])
        xlim = (allZ[:,0].min()*1.05, allZ[:,0].max()*1.05)
        ylim = (allZ[:,1].min()*1.05, allZ[:,1].max()*1.05)
        def _ellipse(ax, pts, color):
            try:
                if pts.shape[0] < 3:
                    return
                import numpy as _np
                from numpy.linalg import eig
                C = _np.cov(pts.T)
                vals, vecs = eig(C)
                order = vals.argsort()[::-1]
                vals, vecs = vals[order], vecs[:,order]
                import matplotlib.patches as mpatches
                theta = _np.degrees(_np.arctan2(vecs[1,0], vecs[0,0]))
                # 2 std dev ellipse
                width, height = 2.0*_np.sqrt(vals[0])*2, 2.0*_np.sqrt(vals[1])*2
                e = mpatches.Ellipse(pts.mean(axis=0), width, height, angle=theta,
                                     edgecolor=color, facecolor='none', lw=1.0, alpha=0.8)
                ax.add_patch(e)
            except Exception:
                pass
        def draw(ax, Z, title):
            for c in np.unique(batch_codes):
                mask = batch_codes == c
                lbl = batch_names[int(c)] if int(c) < len(batch_names) else str(int(c))
                ax.scatter(Z[mask,0], Z[mask,1], s=14, color=cmap(int(c)%cmap.N), alpha=0.85, label=lbl)
                _ellipse(ax, Z[mask], cmap(int(c)%cmap.N))
            ax.set_title(title)
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_xlim(*xlim); ax.set_ylim(*ylim)
        draw(axes[0], Zb, 'Before')
        draw(axes[1], Za, 'After')
        if label_col and label_col in meta.columns:
            labels = meta.loc[common, label_col].astype('category')
            if len(labels.cat.categories) > 1:
                cats = list(labels.cat.categories)
                lower = [c.lower() for c in cats]
                fixed_map = {}
                if 'tumor' in lower and 'normal' in lower:
                    for c in cats:
                        if c.lower() == 'tumor': fixed_map[c] = '^'
                        elif c.lower() == 'normal': fixed_map[c] = 'o'
                markers_cycle = ["o","s","^","D","v","P","X","*","<",">"]
                for i, cat in enumerate(cats):
                    m = fixed_map.get(cat, markers_cycle[i % len(markers_cycle)])
                    mask = labels == cat
                    for ax, Z in ((axes[0], Zb), (axes[1], Za)):
                        ax.scatter(Z[mask,0], Z[mask,1], facecolors='none', edgecolors='k', marker=m, s=65, linewidths=0.9, label=f"{cat} (label)")
        # Consolidated legend
        handles = {}
        for ax in axes:
            for h in ax.collections:
                lbl = h.get_label()
                if lbl and lbl not in handles:
                    handles[lbl] = h
        axes[0].legend(handles.values(), handles.keys(), bbox_to_anchor=(1.02,1), loc='upper left', fontsize='small')
        evr = pca.explained_variance_ratio_
        n_samp = len(common)
        n_gen = len(common_genes)
        fig.suptitle(
            f"PCA Before vs After (HVG {hvg_desc}) — EVR: PC1 {evr[0]*100:.1f}%  PC2 {evr[1]*100:.1f}%  |  n={n_samp} samples, g={n_gen} genes"
        )
        fig.tight_layout()
        fig.savefig(out_path, dpi=160)
        plt.close(fig)
        print(f"[OK] Saved PCA panel: {out_path}")
    except Exception as e:
        print(f"[VIZ] PCA panel failed ({out_path}): {e}")

def parse_args():
    parser = argparse.ArgumentParser(description="Improved VAE Batch Corrector with W&B support.")
    parser.add_argument("--counts", required=True, type=Path)
    parser.add_argument("--metadata", required=True, type=Path)
    parser.add_argument("--out_corrected", required=True, type=Path)
    parser.add_argument("--sample_col", default="sample", type=str)
    parser.add_argument("--batch_col", default="batch", type=str)
    parser.add_argument("--label_col", default=None, type=str)
    parser.add_argument("--genes_in_rows", action="store_true")
    parser.add_argument("--hvg", default=5000, type=int)
    parser.add_argument("--latent_dim", default=32, type=int)
    parser.add_argument("--enc_hidden", default="1024,256", type=str)
    parser.add_argument("--dec_hidden", default="256,1024", type=str)
    parser.add_argument("--adv_hidden", default="128", type=str)
    parser.add_argument("--sup_hidden", default="64", type=str)
    parser.add_argument("--attention_heads", default=4, type=int)
    parser.add_argument("--epochs", default=200, type=int)
    parser.add_argument("--lr", default=1e-3, type=float)
    parser.add_argument("--batch_size", default=256, type=int)
    parser.add_argument("--adv_weight", default=1.0, type=float)
    parser.add_argument("--sup_weight", default=1.0, type=float)
    parser.add_argument("--kl_weight", default=0.001, type=float)
    parser.add_argument("--weight_decay", default=0.0, type=float)
    parser.add_argument("--scheduler", default="cosine_warmup", choices=["none", "plateau", "cosine", "cosine_warmup"], help="LR scheduler (cosine_warmup adds linear warmup + cosine decay)")
    parser.add_argument("--warmup_ratio", default=0.1, type=float, help="Fraction of total optimizer steps used for linear warmup (cosine_warmup only)")
    parser.add_argument("--amp", action="store_true", help="Enable autocast mixed precision")
    parser.add_argument("--precision", default="auto",
                        choices=["auto", "fp16", "bf16", "fp32"],
                        help="Numerical precision mode")
    parser.add_argument("--dropout", default=0.1, type=float)
    parser.add_argument("--patience", default=20, type=int)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--out_latent", type=Path, default=None)
    parser.add_argument("--out_shap", type=Path, default=None)
    parser.add_argument("--save_model_path", type=Path, default="best_model.pt")
    parser.add_argument("--num_workers", default=4, type=int, help="DataLoader workers (increase to improve pipeline overlap)")
    parser.add_argument("--prefetch_factor", default=8, type=int, help="DataLoader prefetch factor (default 2; increase for large GPU)")
    parser.add_argument("--pin_memory", action="store_true", help="Pin memory for faster host->GPU transfer")
    parser.add_argument("--log_latent_every", default=0, type=int, help="Log latent PCA to W&B every N epochs (0=disable)")
    parser.add_argument("--compile", action="store_true", help="Use torch.compile for model (PyTorch 2.x)")
    parser.add_argument("--channels_last", action="store_true", help="Use channels_last memory format for potential speed")
    parser.add_argument("--grad_accum", default=1, type=int, help="Gradient accumulation steps to increase effective batch size")
    parser.add_argument("--log_lr_steps", action="store_true", help="Print / log LR and loss each optimizer step")
    parser.add_argument("--cudnn_benchmark", action="store_true", help="Enable cudnn.benchmark for conv autotuning (may help larger models)")
    parser.add_argument("--fused_optim", action="store_true", help="Use fused AdamW optimizer if available (PyTorch 2.0+)")
    parser.add_argument("--cuda_graphs", action="store_true", help="Capture training step with CUDA Graphs (static shapes; disables grad accumulation)")
    parser.add_argument("--run_id", type=str, default=None, help="Optional external run ID for reproducibility")
    # W&B arguments
    parser.add_argument("--use_wandb", action="store_true", help="Enable W&B logging")
    parser.add_argument("--wandb_project", default="harmonize-nn-vae", type=str, help="W&B project name")
    parser.add_argument("--wandb_entity", default=None, type=str, help="W&B entity (username or team)")
    # Visualisation options
    parser.add_argument("--generate_viz", action="store_true", help="Generate PCA (before/after) and batch boxplots")
    parser.add_argument("--viz_hvg_top", default=2000, type=int, help="Top-N genes for PCA visualisations (0=all)")
    parser.add_argument("--viz_pca_before", default="pca_before.png", type=str, help="Output path for PCA before correction")
    parser.add_argument("--viz_pca_after", default="pca_after.png", type=str, help="Output path for PCA after correction")
    parser.add_argument("--viz_boxplot", default="logCPM_boxplots.png", type=str, help="Output path for logCPM boxplots")
    parser.add_argument("--viz_pca_panel", default="pca_panel.png", type=str, help="Output path for combined before/after PCA panel")
    parser.add_argument("--debug_device", action="store_true", help="Print detailed device / CUDA diagnostic info")
    parser.add_argument("--require_cuda", action="store_true", help="Abort if CUDA not available (helps catch env mismatches in sweeps)")
    parser.add_argument("--fast_start", action="store_true", help="Skip visualisations & latent logging for faster sweeps")
    parser.add_argument("--cache_dir", type=Path, default=None, help="Optional directory to cache logCPM + HVG selection")
    # Overcorrection penalty controls
    parser.add_argument("--cond_min", default=0.10, type=float, help="Minimum acceptable condition separation (silhouette) before penalising overcorrection.")
    parser.add_argument("--overcorr_weight", default=0.5, type=float, help="Penalty weight applied when condition silhouette < cond_min.")
    # Differentiable biology-preserving loss and adaptive adversary scaling
    parser.add_argument("--bio_weight", default=0.0, type=float, help="Weight for supervised center loss on latent (preserve biology)")
    parser.add_argument("--bio_gamma", default=0.5, type=float, help="Relative weight for between-class separation in center loss")
    parser.add_argument("--adaptive_cond_margin", default=0.02, type=float, help="If cond silhouette < cond_min + margin, reduce adversarial strength")
    parser.add_argument("--adaptive_cond_scale", default=0.85, type=float, help="Multiplicative scale on adversarial lambda when condition separation is weak")
    return parser.parse_args()

def select_precision(args):
    if args.precision == "fp32":
        return torch.float32
    if args.precision == "fp16":
        return torch.float16
    if args.precision == "bf16":
        return torch.bfloat16
    # auto
    if torch.cuda.is_available():
        major, _ = torch.cuda.get_device_capability()
        # Ampere (8.x) or newer => prefer bf16
        if major >= 8 and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        else:
            return torch.float16
    return torch.float32

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    precision_dtype = select_precision(args)
    use_amp = args.amp and precision_dtype in (torch.float16, torch.bfloat16)

    if args.cudnn_benchmark:
        torch.backends.cudnn.benchmark = True
    # Global perf knobs
    try:
        torch.set_float32_matmul_precision('high')
    except Exception:
        pass

    import time, hashlib, json
    t0 = time.time()
    # --- W&B Initialization ---
    wandb_run = None
    if args.use_wandb and wandb:
        wandb_run = wandb.init(project=args.wandb_project, entity=args.wandb_entity, config=vars(args))
    
    cuda_ok = torch.cuda.is_available()
    device = torch.device("cuda" if cuda_ok else "cpu")
    print(f"[INFO] Using device: {device}")
    if args.require_cuda and not cuda_ok:
        print("[FATAL] --require_cuda set but CUDA not available. Exiting (likely wrong interpreter or missing GPU drivers).")
        # Provide minimal context even if debug flag not set
        import sys as _sys, os as _os
        print("[FATAL] sys.executable:", _sys.executable)
        print("[FATAL] torch version:", torch.__version__, "torch.version.cuda:", torch.version.cuda)
        print("[FATAL] CUDA_VISIBLE_DEVICES:", _os.environ.get("CUDA_VISIBLE_DEVICES"))
        return
    if args.debug_device:
        import sys, os as _os
        print("[DEBUG] sys.executable:", sys.executable)
        print("[DEBUG] sys.path[0]:", sys.path[0])
        print("[DEBUG] torch version:", torch.__version__, "cuda rt:", torch.version.cuda)
        print("[DEBUG] torch.cuda.is_available():", torch.cuda.is_available())
        print("[DEBUG] torch.cuda.device_count():", torch.cuda.device_count())
        try:
            print("[DEBUG] arch list:", torch.cuda.get_arch_list())
        except Exception as _e:
            print("[DEBUG] get_arch_list error:", _e)
        print("[DEBUG] CUDA_VISIBLE_DEVICES:", _os.environ.get("CUDA_VISIBLE_DEVICES"))
        print("[DEBUG] PATH snippet:", _os.environ.get("PATH", "")[:180], "...")
        print("[DEBUG] Working dir:", _os.getcwd())
        try:
            from torch.backends import cudnn as _cudnn
            print("[DEBUG] cudnn.enabled:", _cudnn.enabled, "version:", _cudnn.version())
        except Exception as _e:
            print("[DEBUG] cudnn info error:", _e)
        try:
            print("[DEBUG] Current device index:", torch.cuda.current_device() if torch.cuda.is_available() else None)
            if torch.cuda.is_available():
                print("[DEBUG] Device name:", torch.cuda.get_device_name(0))
        except Exception as _e:
            print("[DEBUG] device name error:", _e)

   # Load and Preprocess
    t_load0 = time.time()
    counts_raw, meta = load_inputs(args.counts, args.metadata, args.sample_col, args.batch_col, args.label_col, args.genes_in_rows)
    t_load1 = time.time()
    # Optional caching of logCPM + HVG selection to cut startup cost across many sweep runs
    if args.cache_dir:
        args.cache_dir.mkdir(parents=True, exist_ok=True)
        # Create a simple hash of file path + hvg size to identify cache
        h = hashlib.md5(f"{args.counts.resolve()}::{args.hvg}".encode()).hexdigest()[:12]
        logcpm_cache = args.cache_dir / f"logcpm_{h}.npy"
        genes_cache = args.cache_dir / f"hvg_genes_{h}.json"
    else:
        logcpm_cache = genes_cache = None
    if logcpm_cache and logcpm_cache.exists() and genes_cache.exists():
        logcpm_before_full = np.load(logcpm_cache, allow_pickle=False)
        with open(genes_cache) as fh: hvg_genes = json.load(fh)
        logcpm_before_full = pd.DataFrame(logcpm_before_full, index=counts_raw.index, columns=counts_raw.columns)
    else:
        logcpm_before_full = library_size_normalize(counts_raw)
        hvg_genes = select_hvg(logcpm_before_full, args.hvg).columns.tolist()
        if logcpm_cache:
            np.save(logcpm_cache, logcpm_before_full.values, allow_pickle=False)
            with open(genes_cache, 'w') as fh: json.dump(hvg_genes, fh)
    counts_hvg = counts_raw[hvg_genes]
    t_proc = time.time()
    # NOTE: We now train on raw counts (counts_hvg), not standardized counts

    batch_cats = meta[args.batch_col].astype("category")
    batch_idx = batch_cats.cat.codes.values
    if args.label_col:
        # Normalise label strings to avoid hidden whitespace/case issues then categorical-encode
        lbl_series = meta[args.label_col].astype(str).str.strip().str.lower()
        cats = lbl_series.astype("category").cat.categories.tolist()
        label_idx = lbl_series.astype("category").cat.codes.values
        label_classes = cats
    else:
        label_idx, label_classes = (None, None)

    # Datasets and Dataloaders (using raw counts_hvg)
    train_ix, val_ix = _train_val_split_indices(len(counts_hvg), test_size=0.2, seed=args.seed,
                                                label_idx=label_idx, batch_idx=batch_idx)
    ds_train = RNADataset(counts_hvg.values[train_ix], batch_idx[train_ix], label_idx[train_ix] if label_idx is not None else None)
    ds_val = RNADataset(counts_hvg.values[val_ix], batch_idx[val_ix], label_idx[val_ix] if label_idx is not None else None)
    # Only enable pin_memory when CUDA is present (otherwise PyTorch warns and it has no effect)
    effective_pin = args.pin_memory and torch.cuda.is_available()
    if args.pin_memory and not torch.cuda.is_available():
        print("[INFO] Disabling pin_memory (no CUDA device detected).")
    dl_train = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                          pin_memory=effective_pin, persistent_workers=(args.num_workers>0), prefetch_factor=args.prefetch_factor)
    dl_val = DataLoader(ds_val, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                        pin_memory=effective_pin, persistent_workers=(args.num_workers>0), prefetch_factor=args.prefetch_factor)

    # Initialize and Train Model
    model = VaeAttentionBatchCorrector(
        n_genes=counts_hvg.shape[1], latent_dim=args.latent_dim,
        enc_hidden=tuple(map(int, args.enc_hidden.split(","))), dec_hidden=tuple(map(int, args.dec_hidden.split(","))),
        adv_hidden=tuple(map(int, args.adv_hidden.split(","))), sup_hidden=tuple(map(int, args.sup_hidden.split(","))),
        n_batches=len(batch_cats.cat.categories), n_labels=len(label_classes) if label_classes else None,
        dropout=args.dropout, attention_heads=args.attention_heads)

    if args.fast_start:
        # Override heavy options for speed
        if args.generate_viz:
            print("[FAST] Disabling visualisations for fast_start")
        args.generate_viz = False
        if args.log_latent_every != 0:
            print("[FAST] Disabling latent logging for fast_start")
        args.log_latent_every = 0
    if args.channels_last:
        # channels_last only meaningful for 4D tensors (Conv nets). This model is MLP-based (2D inputs).
        # We keep flag for parity, but skip to avoid runtime errors.
        print("[INFO] --channels_last requested but model inputs are 2D; skipping (no benefit for linear layers).")

    if args.compile:
        can_compile = True
        # Triton often unavailable / limited on Windows; skip gracefully
        try:
            import triton  # type: ignore  # noqa: F401
        except Exception:
            can_compile = False
            print("[WARN] Triton not found; skipping torch.compile (install 'triton' to enable).")
        if os.name == 'nt' and not can_compile:
            pass  # already warned
        elif can_compile:
            try:
                model = torch.compile(model, mode="reduce-overhead")
                print("[INFO] torch.compile enabled")
            except Exception as e:
                print(f"[WARN] torch.compile failed: {e}")
    if wandb_run: wandb_run.watch(model, log="all", log_freq=100)
    
    fit = train_model(model, dl_train, dl_val, epochs=args.epochs, lr=args.lr, weight_decay=args.weight_decay,
                      adv_weight=args.adv_weight, sup_weight=args.sup_weight, kl_weight=args.kl_weight,
                      use_amp=args.amp, scheduler_type=args.scheduler, patience=args.patience,
                      save_best_path=args.save_model_path, wandb_run=wandb_run, batch_classes=batch_cats.cat.categories.tolist(), device=device,
                      log_latent_every=args.log_latent_every, grad_accum_steps=args.grad_accum, warmup_ratio=args.warmup_ratio,
                      log_lr_steps=args.log_lr_steps, use_fused_optim=args.fused_optim, use_cuda_graphs=args.cuda_graphs,
                      precision_dtype=precision_dtype,
                      cond_min=args.cond_min, overcorr_weight=args.overcorr_weight,
                      adaptive_cond_margin=getattr(args, 'adaptive_cond_margin', 0.02), adaptive_cond_scale=getattr(args, 'adaptive_cond_scale', 0.85),
                      bio_weight=getattr(args, 'bio_weight', 0.0), bio_gamma=getattr(args, 'bio_gamma', 0.5))
    model = fit["model"]
    t_train_end = time.time()
    print(f"[TIMING] load={t_load1-t_load0:.2f}s preprocess={t_proc-t_load1:.2f}s train_total={t_train_end-t_proc:.2f}s")

    # Inference and Output
    model.eval()
    with torch.no_grad():
        # Move the full dataset to the GPU for inference
        X_all = torch.tensor(counts_hvg.values, dtype=torch.float32).to(device, non_blocking=True)
        recon_mu, _, _, _, _, _, z_lat = model(X_all)
        corrected_counts = recon_mu.cpu().numpy()
        z_lat_cpu = z_lat.cpu().numpy()
        
    corrected_df = pd.DataFrame(corrected_counts, index=counts_hvg.index, columns=counts_hvg.columns)
    corrected_df.to_csv(args.out_corrected)
    print(f"[OK] Wrote corrected matrix to: {args.out_corrected}")

    if args.out_latent:
        z_df = pd.DataFrame(z_lat_cpu, index=counts_hvg.index, columns=[f"z{i+1}" for i in range(z_lat_cpu.shape[1])])
        z_df.to_csv(args.out_latent)
        print(f"[OK] Wrote latent embedding to: {args.out_latent}")

    # SHAP Interpretability
    if args.out_shap and shap is not None:
        print("\n[INFO] Starting SHAP analysis on GPU...")
        # Move SHAP background and test samples to the GPU
        background = torch.tensor(counts_hvg.values[train_ix[:100]], dtype=torch.float32).to(device) # <-- CHANGED
        test_samples = torch.tensor(counts_hvg.values[val_ix[:20]], dtype=torch.float32).to(device) # <-- CHANGED

        # This wrapper function ensures data passed by SHAP is moved to the GPU
        def adv_predictor(x):
            return model(x.to(device))[4] # index 4 is batch_logits # <-- CHANGED
        
        explainer = shap.DeepExplainer(adv_predictor, background)
        shap_values = explainer.shap_values(test_samples)
        shap_df = pd.DataFrame({'gene': counts_hvg.columns, 'shap_importance': np.abs(shap_values).mean(axis=(0,1))})
        shap_df.sort_values(by='shap_importance', ascending=False, inplace=True)
        print(f"\n--- Top 10 Genes Driving Batch Effects (SHAP) ---\n{shap_df.head(10)}")
        shap_df.to_csv(args.out_shap, index=False)
        print(f"[OK] Saved SHAP importance to: {args.out_shap}")

    # Visualisations
    if args.generate_viz:
        try:
            # Before = logCPM normalised from raw counts; After = corrected (already on positive scale from dec_mu Softplus)
            # logcpm_before_full already computed (maybe cached)
            corrected_read = corrected_df.copy()
            # Bring corrected into logCPM-like space for fair comparison if values are counts-scale
            if (corrected_read.values < 0).sum() == 0:
                # add 1 then log transform
                corrected_log = np.log1p(corrected_read)
            else:
                corrected_log = corrected_read  # assume already logCPM
            if args.viz_hvg_top and args.viz_hvg_top > 0:
                vars_before = logcpm_before_full.var(axis=0)
                top_genes = vars_before.nlargest(min(args.viz_hvg_top, vars_before.shape[0])).index
                before_sel = logcpm_before_full.loc[:, top_genes]
                after_sel = corrected_log.loc[:, [g for g in top_genes if g in corrected_log.columns]]
                if after_sel.shape[1] < 2:
                    after_sel = corrected_log
                    before_sel = logcpm_before_full
            else:
                before_sel = logcpm_before_full
                after_sel = corrected_log
            _pca_plot(before_sel, meta, args.batch_col, args.label_col, Path(args.viz_pca_before), title=f"PCA (before) HVG={args.viz_hvg_top}")
            _pca_plot(after_sel, meta, args.batch_col, args.label_col, Path(args.viz_pca_after), title=f"PCA (after) HVG={args.viz_hvg_top}")
            if args.viz_pca_panel:
                _pca_panel(before_sel, after_sel, meta, args.batch_col, args.label_col, Path(args.viz_pca_panel), hvg_desc=str(args.viz_hvg_top))
            # Boxplots use same gene set intersection
            common_samples = logcpm_before_full.index.intersection(corrected_log.index)
            _boxplots(logcpm_before_full.loc[common_samples], corrected_log.loc[common_samples], meta.loc[common_samples], args.batch_col, Path(args.viz_boxplot))
        except Exception as e:
            print(f"[VIZ] Failed to generate visualisations: {e}")

    # --- W&B Artifact Logging ---
    if wandb_run:
        print("[INFO] Logging outputs as W&B Artifacts...")
        artifact = wandb.Artifact(name=f"{wandb_run.id}-outputs", type="dataset")
        artifact.add_file(args.out_corrected)
        if args.out_latent: artifact.add_file(args.out_latent)
        if args.save_model_path: artifact.add_file(args.save_model_path)
        if args.out_shap and os.path.exists(args.out_shap): artifact.add_file(args.out_shap)
        # Include visualisations if generated
        try:
            if args.generate_viz:
                for p in [args.viz_pca_before, args.viz_pca_after, args.viz_pca_panel, args.viz_boxplot]:
                    if p and isinstance(p, (str, os.PathLike)) and os.path.exists(p):
                        artifact.add_file(p)
        except Exception as _e:
            print(f"[WARN] Unable to attach viz files to artifact: {_e}")
        wandb_run.log_artifact(artifact)
        wandb_run.finish()

if __name__ == "__main__":
    main()
