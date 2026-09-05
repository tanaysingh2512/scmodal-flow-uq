"""
Conditional flow matching models for cross-modal protein imputation.

Two variants:
  - VelocityNet: the base flow (point-prediction velocity field, MSE loss).
    Uncertainty comes from sample-spread across K stochastic draws.
  - HeteroscedasticVelocityNet (UA-Flow, Stretch Goal 2): adds a log-variance
    head trained with Gaussian NLL, giving a *learned* uncertainty estimate
    directly instead of relying only on sample-spread.

Both are conditioned on a frozen encoder latent z (from scMODAL's E_A/E_B).
"""

import torch
import torch.nn as nn


class VelocityNet(nn.Module):
    """Base conditional flow-matching velocity field."""

    def __init__(self, feat_dim: int, latent_dim: int, hidden: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feat_dim + 1 + latent_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, feat_dim),
        )

    def forward(self, y_t, t, z):
        t = t.view(-1, 1)
        return self.net(torch.cat([y_t, t, z], dim=1))


def flow_loss(vnet: VelocityNet, y, z):
    eps = torch.randn_like(y)
    t = torch.rand(y.size(0), device=y.device)
    y_t = (1 - t).view(-1, 1) * eps + t.view(-1, 1) * y
    target_v = y - eps
    pred_v = vnet(y_t, t, z)
    return ((pred_v - target_v) ** 2).mean()


def train_flow(z_train, y_train, feat_dim, latent_dim, batch_size=256, lr=1e-3, n_steps=5000):
    """Trains the base VelocityNet. Matches the original step-based schedule
    (n_steps=5000, batch_size=256, lr=1e-3)."""
    vnet = VelocityNet(feat_dim, latent_dim)
    optimizer = torch.optim.Adam(vnet.parameters(), lr=lr)

    N = y_train.shape[0]
    vnet.train()
    for step in range(n_steps):
        idx = torch.randint(0, N, (batch_size,))
        loss = flow_loss(vnet, y_train[idx], z_train[idx])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % 500 == 0:
            print(f"step {step}, flow_loss = {loss.item():.4f}")
    return vnet


@torch.no_grad()
def sample_flow(vnet: VelocityNet, z, feat_dim, steps=100, K=50):
    """Draws K stochastic samples per cell via Euler integration of the
    learned velocity field. Sample-spread across K is the uncertainty
    signal for the base flow."""
    vnet.eval()
    z_rep = z.repeat_interleave(K, dim=0)
    y = torch.randn(z_rep.size(0), feat_dim)
    for i in range(steps):
        t = torch.full((y.size(0),), i / steps)
        pred_v = vnet(y, t, z_rep)
        y = y + pred_v * (1.0 / steps)
    return y.view(z.size(0), K, feat_dim)


class HeteroscedasticVelocityNet(nn.Module):
    """Same architecture as VelocityNet, plus a log-variance head
    (Stretch Goal 2 -- UA-Flow)."""

    def __init__(self, feat_dim: int, latent_dim: int, hidden: int = 512):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(feat_dim + 1 + latent_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
        )
        self.mean_head = nn.Linear(hidden, feat_dim)
        self.logvar_head = nn.Linear(hidden, feat_dim)

    def forward(self, y_t, t, z):
        t = t.view(-1, 1)
        h = self.shared(torch.cat([y_t, t, z], dim=1))
        return self.mean_head(h), self.logvar_head(h)


def heteroscedastic_flow_loss(vnet: HeteroscedasticVelocityNet, y, z):
    eps = torch.randn_like(y)
    t = torch.rand(y.size(0), device=y.device)
    y_t = (1 - t).view(-1, 1) * eps + t.view(-1, 1) * y
    target_v = y - eps

    pred_v, pred_logvar = vnet(y_t, t, z)
    pred_var = torch.exp(pred_logvar).clamp(min=1e-6)
    nll = 0.5 * (pred_logvar + (pred_v - target_v) ** 2 / pred_var)
    return nll.mean()


def train_ua_flow(z_train, y_train, feat_dim, latent_dim, epochs=200, batch_size=256, lr=1e-3, n_steps=5000):
    """Trains the heteroscedastic UA-Flow. Matches the base flow's
    step-based schedule (n_steps=5000, batch_size=256, lr=1e-3).

    `epochs` is part of the original notebook's function signature (cell 139)
    but is never actually used in its body either -- training there is purely
    step-based via `n_steps`. Kept here unused, matching the original exactly,
    rather than silently dropping it (found during the reproducibility audit;
    see decisions.md). Do not assume passing `epochs=` changes anything.
    """
    vnet = HeteroscedasticVelocityNet(feat_dim, latent_dim)
    optimizer = torch.optim.Adam(vnet.parameters(), lr=lr)

    N = z_train.shape[0]
    vnet.train()
    for step in range(n_steps):
        idx = torch.randint(0, N, (batch_size,))
        loss = heteroscedastic_flow_loss(vnet, y_train[idx], z_train[idx])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % 500 == 0:
            print(f"step {step}, NLL loss = {loss.item():.4f}")
    return vnet


@torch.no_grad()
def sample_ua_flow(vnet: HeteroscedasticVelocityNet, z, feat_dim, steps=100, K=50):
    """Draws K stochastic samples per cell (same sample-spread signal as the
    base flow) -- used for comparison against the learned-variance head."""
    vnet.eval()
    z_rep = z.repeat_interleave(K, dim=0)
    y = torch.randn(z_rep.size(0), feat_dim)
    for i in range(steps):
        t = torch.full((y.size(0),), i / steps)
        pred_v, _ = vnet(y, t, z_rep)
        y = y + pred_v * (1.0 / steps)
    return y.view(z.size(0), K, feat_dim)


@torch.no_grad()
def learned_variance(vnet: HeteroscedasticVelocityNet, y, z):
    """Single forward pass at t=1 (endpoint) to read off the learned
    per-feature std, no sampling needed."""
    vnet.eval()
    t_end = torch.ones(z.size(0))
    _, pred_logvar = vnet(y, t_end, z)
    return torch.exp(0.5 * pred_logvar)
