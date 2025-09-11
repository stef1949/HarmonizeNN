import math
import torch

from vae_attention_model import NegativeBinomialLoss, reparameterize, VaeAttentionBatchCorrector


def _nb_loglik(mu, theta, x, eps=1e-8):
    # Matches implementation formula; returns negative log-likelihood per element
    log_theta_mu = torch.log(theta + mu + eps)
    res = (
        theta * (torch.log(theta + eps) - log_theta_mu)
        + x * (torch.log(mu + eps) - log_theta_mu)
        + torch.lgamma(x + theta)
        - torch.lgamma(theta)
        - torch.lgamma(x + 1.0)
    )
    return -res


def test_negative_binomial_loss_matches_formula():
    torch.manual_seed(0)
    x = torch.tensor([[0.0, 1.0, 3.0], [2.0, 0.0, 4.0]])
    mu = torch.tensor([[0.5, 1.2, 3.5], [2.5, 0.3, 4.2]])
    theta = torch.tensor([[0.8, 0.9, 1.1], [0.7, 0.6, 1.3]])
    preds = {"mu": mu, "theta": theta}

    loss = NegativeBinomialLoss(reduction="mean")(preds, x)
    manual = _nb_loglik(mu, theta, x).mean()
    assert torch.allclose(loss, manual, rtol=1e-6, atol=1e-6)


def test_reparameterize_zero_variance_returns_mu():
    mu = torch.randn(4, 3)
    # Very negative logvar -> std ~ 0
    logvar = torch.full_like(mu, -1e20)
    z = reparameterize(mu, logvar)
    assert torch.allclose(z, mu, atol=0.0, rtol=0.0)


@torch.no_grad()
def test_vae_attention_forward_shapes_and_positivity():
    torch.manual_seed(0)
    B, G = 5, 10
    x = torch.rand(B, G)
    batch_idx = torch.tensor([0, 1, 0, 1, 0])
    # Test all dispersion modes quickly
    for disp in ("gene", "global", "gene-batch"):
        model = VaeAttentionBatchCorrector(
            num_genes=G,
            num_batches=2,
            latent_dim=8,
            hidden_dim=16,
            attention_dim=8,
            n_heads=2,
            dropout=0.0,
            dispersion=disp,
        )
        out = model(x, batch_idx)
        mu, theta = out["mu"], out["theta"]
        assert mu.shape == (B, G)
        # theta should broadcast or match (B,G)
        if disp == "gene":
            assert theta.shape == (B, G)
        elif disp == "global":
            assert theta.numel() == 1 or theta.shape == (B, G)
        else:  # gene-batch
            assert theta.shape == (B, G)
        assert torch.all(mu > 0)
        assert torch.all(theta > 0)

        losses = model.vae_loss(out, x, beta=0.1)
        assert set(losses.keys()) == {"total", "recon", "kl"}
        for k in losses:
            v = losses[k]
            assert torch.is_tensor(v) and v.ndim == 0 and torch.isfinite(v)

