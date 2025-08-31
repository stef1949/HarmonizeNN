import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict


class NegativeBinomialLoss(nn.Module):
    """Negative Binomial log-likelihood loss for (approx) count data.
    Expects preds dict with keys mu (>0) and theta (>0). Target are non-negative values.
    """
    def __init__(self, reduction: str = "mean", eps: float = 1e-8, clamp_theta_min: float = 1e-4, clamp_mu_min: float = 1e-5):
        super().__init__()
        self.reduction = reduction
        self.eps = eps
        self.clamp_theta_min = clamp_theta_min
        self.clamp_mu_min = clamp_mu_min

    def forward(self, preds: Dict[str, torch.Tensor], target: torch.Tensor):  # type: ignore
        mu = preds["mu"].clamp(min=self.clamp_mu_min)
        theta = preds["theta"].clamp(min=self.clamp_theta_min)
        t = target
        log_theta_mu = torch.log(theta + mu + self.eps)
        res = (
            theta * (torch.log(theta + self.eps) - log_theta_mu)
            + t * (torch.log(mu + self.eps) - log_theta_mu)
            + torch.lgamma(t + theta)
            - torch.lgamma(theta)
            - torch.lgamma(t + 1.0)
        )
        nll = -res
        if self.reduction == "mean":
            return nll.mean()
        if self.reduction == "sum":
            return nll.sum()
        return nll


def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std


class GeneTokenSelfAttention(nn.Module):
    """Self-attention treating each gene as a token (simple transformer block)."""
    def __init__(self, num_genes: int, d_model: int = 128, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.gene_embed = nn.Linear(1, d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.ReLU(),
            nn.Linear(d_model * 2, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.out_proj = nn.Linear(d_model, 1)
        self.num_genes = num_genes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, G)
        b, g = x.shape
        x_tok = x.view(b, g, 1)
        h = self.gene_embed(x_tok)
        attn_out, _ = self.attn(h, h, h)
        h = self.norm1(h + attn_out)
        ff_out = self.ff(h)
        h = self.norm2(h + ff_out)
        out = self.out_proj(h).view(b, g)
        return out


class VaeAttentionBatchCorrector(nn.Module):
    """VAE with gene-level self-attention and optional dispersion parameterisation.

    Dispersion modes:
      gene: one theta per gene
      global: single shared theta
      gene-batch: theta per (batch, gene)
    """
    def __init__(
        self,
        num_genes: int,
        num_batches: int,
        latent_dim: int = 32,
        hidden_dim: int = 256,
        attention_dim: int = 128,
        n_heads: int = 4,
        dropout: float = 0.1,
        dispersion: str = "gene",
    ):
        super().__init__()
        self.num_genes = num_genes
        self.num_batches = num_batches
        self.latent_dim = latent_dim
        self.dispersion = dispersion

        self.attention = GeneTokenSelfAttention(num_genes, d_model=attention_dim, n_heads=n_heads, dropout=dropout)

        self.encoder = nn.Sequential(
            nn.Linear(num_genes, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.enc_mu = nn.Linear(hidden_dim, latent_dim)
        self.enc_logvar = nn.Linear(hidden_dim, latent_dim)
        self.batch_embed = nn.Embedding(num_batches, latent_dim)

        self.decoder_hidden = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.dec_mu = nn.Linear(hidden_dim, num_genes)

        if dispersion == "gene":
            self.theta = nn.Parameter(torch.randn(num_genes))
        elif dispersion == "global":
            self.theta = nn.Parameter(torch.randn(1))
        elif dispersion == "gene-batch":
            self.theta = nn.Parameter(torch.randn(num_batches, num_genes))
        else:
            raise ValueError("Invalid dispersion mode")

    def get_theta(self, batch_idx: torch.Tensor) -> torch.Tensor:
        if self.dispersion == "gene":
            return F.softplus(self.theta)
        if self.dispersion == "global":
            return F.softplus(self.theta)
        if self.dispersion == "gene-batch":
            return F.softplus(self.theta[batch_idx])
        raise RuntimeError

    def forward(self, x: torch.Tensor, batch_idx: torch.Tensor):
        # x: (B,G) pseudo-counts/CPM (non-negative)
        attn_feats = self.attention(x)
        h = self.encoder(attn_feats)
        mu_z = self.enc_mu(h)
        logvar_z = self.enc_logvar(h)
        z = reparameterize(mu_z, logvar_z)
        z = z + self.batch_embed(batch_idx)
        dec_h = self.decoder_hidden(z)
        mu = F.softplus(self.dec_mu(dec_h))  # ensure positive
        theta = self.get_theta(batch_idx)
        if theta.dim() == 1:
            theta_exp = theta.unsqueeze(0).expand_as(mu)
        else:
            theta_exp = theta
        return {"mu": mu, "theta": theta_exp, "z": z, "z_mu": mu_z, "z_logvar": logvar_z}

    def vae_loss(self, out: Dict[str, torch.Tensor], x: torch.Tensor, beta: float = 1.0):
        nb_loss = NegativeBinomialLoss()
        recon = nb_loss({"mu": out["mu"], "theta": out["theta"]}, x)
        kl = -0.5 * torch.mean(1 + out["z_logvar"] - out["z_mu"] ** 2 - out["z_logvar"].exp())
        total = recon + beta * kl
        return {"total": total, "recon": recon, "kl": kl}

    @torch.no_grad()
    def reconstruct(self, x: torch.Tensor, batch_idx: torch.Tensor):
        out = self.forward(x, batch_idx)
        return out["mu"], out["z"]
