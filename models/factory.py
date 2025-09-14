#!/usr/bin/env python3
"""
Model factory helpers.

Centralizes model construction for AE vs VAE+attention variants while keeping
the original CLI behavior identical.
"""
from __future__ import annotations

from typing import Optional

from .ae import AEBatchCorrector

try:
    # Optional dependency present in the repo root
    from vae_attention_model import VaeAttentionBatchCorrector  # type: ignore
except Exception:
    VaeAttentionBatchCorrector = None  # type: ignore


def build_model_from_args(args, n_genes: int, n_batches: int, n_labels: Optional[int]):
    """Return a model instance based on parsed CLI args.

    Parameters
    - args: argparse.Namespace from NN_batch_correct.py
    - n_genes: number of input genes (features)
    - n_batches: number of batch classes
    - n_labels: number of label classes (or None)
    """
    model_type = getattr(args, "model_type", "ae")
    if model_type == "vae_attention":
        if VaeAttentionBatchCorrector is None:
            raise RuntimeError("vae_attention model requested but module not available.")
        return VaeAttentionBatchCorrector(
            num_genes=n_genes,
            num_batches=n_batches,
            latent_dim=args.latent_dim,
            hidden_dim=args.vae_hidden_dim,
            attention_dim=args.vae_attention_dim,
            n_heads=args.vae_attn_heads,
            dropout=args.dropout,
            dispersion=args.vae_dispersion,
            attn_max_tokens=args.attn_max_tokens,
        )

    # Default AE path
    enc_hidden = tuple(int(x) for x in args.enc_hidden.split(",") if x.strip())
    dec_hidden = tuple(int(x) for x in args.dec_hidden.split(",") if x.strip())
    adv_hidden = tuple(int(x) for x in args.adv_hidden.split(",") if x.strip())
    sup_hidden = tuple(int(x) for x in args.sup_hidden.split(",") if x.strip())
    return AEBatchCorrector(
        n_genes=n_genes,
        latent_dim=args.latent_dim,
        enc_hidden=enc_hidden,
        dec_hidden=dec_hidden,
        adv_hidden=adv_hidden,
        sup_hidden=sup_hidden,
        n_batches=n_batches,
        n_labels=n_labels,
        dropout=args.dropout,
        adv_lambda=args.adv_weight,
        use_residual=args.use_residual,
    )

