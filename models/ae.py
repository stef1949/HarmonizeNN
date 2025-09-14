#!/usr/bin/env python3
"""
Adversarial autoencoder components used by NN_batch_correct.py

This module extracts the model building blocks from the training script to
improve modularity and reuse without changing public behavior.
"""
from __future__ import annotations

import torch
import torch.nn as nn


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


class ResidualBlock(nn.Module):
    """
    Residual block with skip connection: x + FFN(x) + LayerNorm
    """

    def __init__(self, hidden_size: int, dropout: float = 0.0):
        super().__init__()
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.Dropout(dropout),
        )
        self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(self, x):
        return self.layer_norm(x + self.ffn(x))


def make_mlp(sizes, dropout=0.0, last_activation=None, use_residual=False):
    """
    Hidden layers: Linear -> LayerNorm -> SiLU -> Dropout
    Output layer: optional activation per arg

    If use_residual=True, uses ResidualBlock layers instead.
    Note: For residual connections to work, all hidden layer sizes must be the same.
    """
    if use_residual:
        if len(sizes) < 3:
            raise ValueError(
                "Residual networks need at least input, hidden, and output layers"
            )

        # Check that all hidden layers have the same size for residual connections
        hidden_sizes = sizes[1:-1]
        if len(set(hidden_sizes)) > 1:
            raise ValueError(
                f"For residual connections, all hidden layer sizes must be the same. Got: {hidden_sizes}"
            )

        hidden_size = hidden_sizes[0]
        num_hidden_layers = len(hidden_sizes)

        layers = []

        # Input projection to hidden size
        layers.append(nn.Linear(sizes[0], hidden_size))
        layers.extend([nn.LayerNorm(hidden_size), nn.SiLU(), nn.Dropout(dropout)])

        # Residual blocks
        for _ in range(num_hidden_layers):
            layers.append(ResidualBlock(hidden_size, dropout))

        # Output layer
        layers.append(nn.Linear(hidden_size, sizes[-1]))
        if last_activation == "relu":
            layers.append(nn.ReLU())
        elif last_activation == "tanh":
            layers.append(nn.Tanh())
        elif last_activation == "sigmoid":
            layers.append(nn.Sigmoid())

        return nn.Sequential(*layers)

    else:
        # Original implementation
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
        n_labels: int | None = None,
        dropout: float = 0.1,
        adv_lambda: float = 1.0,
        use_residual: bool = False,
    ):
        super().__init__()
        self.n_labels = n_labels
        self.grl = GradReverseLayer(lambda_=adv_lambda)

        enc_sizes = [n_genes] + list(enc_hidden) + [latent_dim]
        dec_sizes = [latent_dim] + list(dec_hidden) + [n_genes]
        self.encoder = make_mlp(
            enc_sizes, dropout=dropout, last_activation=None, use_residual=use_residual
        )
        self.decoder = make_mlp(
            dec_sizes, dropout=dropout, last_activation=None, use_residual=use_residual
        )

        adv_sizes = [latent_dim] + list(adv_hidden) + [n_batches]
        # Keep simple for adversarial head (no residuals)
        self.adv = make_mlp(
            adv_sizes, dropout=dropout, last_activation=None, use_residual=False
        )

        if n_labels is not None:
            sup_sizes = [latent_dim] + list(sup_hidden) + [n_labels]
            # Keep simple for supervised head (no residuals)
            self.sup = make_mlp(
                sup_sizes, dropout=dropout, last_activation=None, use_residual=False
            )
        else:
            self.sup = None

    def forward(self, x, adv_lambda: float | None = None):
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

