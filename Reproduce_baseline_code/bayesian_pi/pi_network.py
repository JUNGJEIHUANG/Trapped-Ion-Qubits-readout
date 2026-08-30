"""Permutation-invariant network (PI-Network) of Zhou et al., PRL 137, 013601 (2026).

Architecture, from the right panel of the paper's Fig. 2:

    input   : log-likelihood ratios s_i = log[f(n_i) / g(n_i)]
              plus auxiliary parameters (N, theta_g, theta_f)
    encoder : Phi, permutation-invariant over {s_i}
    head    : Pi, a fully connected network
    output  : P_net(l | {n_i}) over a discrete l-grid, normalised by softmax

Trained by minimising the KL divergence to the exact Bayesian posterior over a
broad sampling of (theta_f, theta_g, l, N) covering the experimental range.

The permutation-invariance is not decoration: the exact posterior of Eq. (2)
depends on {n_i} only through the sum of per-shot log-likelihood contributions,
so a Deep Sets encoder (Zaheer et al. 2017, cited by the paper as [41]) with a
sum pooling can represent it exactly. That is why the network can be trained to
match the posterior rather than merely approximate it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

__all__ = ["PINetworkConfig", "PINetwork", "make_training_batch"]


@dataclass
class PINetworkConfig:
    n_grid: int = 101
    element_hidden: int = 128
    embed_dim: int = 128
    head_hidden: int = 256
    n_element_layers: int = 3
    n_head_layers: int = 3
    n_aux: int = 5  # log N, alpha_g, beta_g, alpha_f, beta_f


def _mlp(sizes, out_act=None) -> nn.Sequential:
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            layers.append(nn.GELU())
    if out_act is not None:
        layers.append(out_act)
    return nn.Sequential(*layers)


class PINetwork(nn.Module):
    """Deep-Sets encoder Phi followed by a fully connected head Pi."""

    def __init__(self, config: PINetworkConfig | None = None):
        super().__init__()
        self.config = cfg = config or PINetworkConfig()

        elem_sizes = [1] + [cfg.element_hidden] * (cfg.n_element_layers - 1) + [cfg.embed_dim]
        self.phi = _mlp(elem_sizes)

        # Both sum and mean pooling are kept: sum carries the extensive part of
        # the log-likelihood (which is what Eq. (2) accumulates over shots),
        # mean carries the intensive part and keeps the input scale bounded as
        # N grows over the two orders of magnitude the paper sweeps.
        head_in = 2 * cfg.embed_dim + cfg.n_aux
        head_sizes = [head_in] + [cfg.head_hidden] * (cfg.n_head_layers - 1) + [cfg.n_grid]
        self.pi = _mlp(head_sizes)

        self.register_buffer("l_grid", torch.linspace(0.0, 1.0, cfg.n_grid))

    def forward(self, s: torch.Tensor, aux: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """Return log P_net(l | {n_i}) of shape ``(B, n_grid)``.

        Parameters
        ----------
        s : (B, N) log-likelihood ratios.
        aux : (B, n_aux) auxiliary parameters.
        mask : (B, N) bool, ``True`` for valid elements. Enables ragged N in a
            batch without letting padding contribute to the pooled statistics.
        """
        h = self.phi(s.unsqueeze(-1))  # (B, N, embed)
        if mask is None:
            pooled_sum = h.sum(dim=1)
            pooled_mean = h.mean(dim=1)
        else:
            m = mask.unsqueeze(-1).to(h.dtype)
            counts = m.sum(dim=1).clamp_min(1.0)
            pooled_sum = (h * m).sum(dim=1)
            pooled_mean = pooled_sum / counts
        # Sum pooling grows linearly in N; normalising keeps the head's input
        # in a fixed range while the raw scale is restored through log N in aux.
        pooled_sum = pooled_sum / s.shape[1]
        logits = self.pi(torch.cat([pooled_sum, pooled_mean, aux], dim=-1))
        return torch.log_softmax(logits, dim=-1)

    @torch.no_grad()
    def posterior(self, s: np.ndarray, aux: np.ndarray) -> np.ndarray:
        self.eval()
        device = next(self.parameters()).device
        st = torch.as_tensor(np.atleast_2d(s), dtype=torch.float32, device=device)
        at = torch.as_tensor(np.atleast_2d(aux), dtype=torch.float32, device=device)
        log_p = self(st, at)
        # Convert a discrete distribution over the grid into a density, so it is
        # on the same footing as the analytic posterior in bayes_em.
        dl = float(self.l_grid[1] - self.l_grid[0])
        return (log_p.exp() / dl).cpu().numpy()


def make_training_batch(
    batch_size: int,
    n_shots: int,
    rng: np.random.Generator,
    alpha_range=(0.5, 50.0),
    beta_range=(0.05, 6.0),
    n_grid: int = 101,
):
    """Sample (s, aux, target posterior) triples for KL training.

    Parameter ranges are the "sufficiently broad sampling of (theta_f, theta_g,
    l, N) covering the experimental range" the paper specifies. The defaults
    here span mean counts from well below 1 to ~300 photons/ROI, bracketing the
    paper's 2 to 138 photons/shot regime; narrow them to the register at hand
    once real calibration data is available.
    """
    from .bayes_em import posterior_over_l
    from .distributions import SuperPoissonian, Theta

    l_grid = np.linspace(0.0, 1.0, n_grid)
    S = np.empty((batch_size, n_shots), dtype=np.float32)
    AUX = np.empty((batch_size, 5), dtype=np.float32)
    TGT = np.empty((batch_size, n_grid), dtype=np.float32)

    for b in range(batch_size):
        theta_g = Theta(rng.uniform(*alpha_range), rng.uniform(*beta_range))
        theta_f = Theta(rng.uniform(*alpha_range), rng.uniform(*beta_range))
        l_true = float(rng.uniform(0.0, 1.0))

        bright = rng.random(n_shots) < l_true
        counts = np.where(
            bright,
            SuperPoissonian.sample(theta_f, n_shots, rng),
            SuperPoissonian.sample(theta_g, n_shots, rng),
        ).astype(float)

        f = SuperPoissonian.pmf(counts, theta_f)
        g = SuperPoissonian.pmf(counts, theta_g)
        S[b] = np.log(np.clip(f, 1e-300, None)) - np.log(np.clip(g, 1e-300, None))
        AUX[b] = [np.log(n_shots), theta_g.alpha, theta_g.beta, theta_f.alpha, theta_f.beta]

        post = posterior_over_l(counts, theta_f, theta_g, l_grid)
        dl = float(l_grid[1] - l_grid[0])
        p = post * dl
        TGT[b] = p / max(p.sum(), 1e-300)

    return S, AUX, TGT


class SitePINetwork(nn.Module):
    """Permutation-EQUIvariant PI-Network over the sites of a single frame.

    ADAPTATION of Zhou et al.'s PI-Network for the Site-AIT protocol, and the
    only part of that paper that has to be re-derived rather than transcribed.

    In the source paper the network is permutation-*invariant* over repeated
    shots and returns one posterior over the scalar occupation ``l``. Site-AIT
    reads out 300 sites from one frame, so the required symmetry is different:
    relabelling the sites must permute the outputs, not leave them fixed. The
    network is therefore permutation-*equivariant* over sites, built from the
    same Deep Sets ingredients the paper cites -- a shared per-element encoder
    plus a pooled register-level context that every site reads back.

    Why this is the fair strong version rather than an embellishment: without
    it, the method reduces per frame to an independent per-site likelihood
    ratio on a scalar ROI sum, which cannot represent crosstalk at all. Giving
    it a register-level channel is what makes the comparison against Site-AIT's
    cross-site attention a comparison of *mechanisms* rather than a comparison
    of "has global context" against "does not".

    Input per site is the LLR ``s_k = log[f(n_k)/g(n_k)]`` plus that site's
    ``(alpha_g, beta_g, alpha_f, beta_f)``, so the network consumes exactly the
    quantities the paper's Bayesian machinery produces -- nothing from the
    image that the source method does not already use.
    """

    def __init__(self, n_features: int = 5, hidden: int = 128, embed: int = 128, n_blocks: int = 2):
        super().__init__()
        self.encoder = _mlp([n_features, hidden, embed])
        # Each block: pool over sites, broadcast the context back, refine.
        self.blocks = nn.ModuleList(
            [_mlp([2 * embed, hidden, embed]) for _ in range(n_blocks)]
        )
        self.head = _mlp([embed, hidden, 1])

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        """``feats`` is ``(B, n_sites, n_features)``; returns logits ``(B, n_sites)``."""
        h = self.encoder(feats)
        for block in self.blocks:
            context = h.mean(dim=1, keepdim=True).expand_as(h)
            h = h + block(torch.cat([h, context], dim=-1))
        return self.head(h).squeeze(-1)

    @torch.no_grad()
    def predict_proba(self, feats: np.ndarray, batch: int = 256) -> np.ndarray:
        self.eval()
        device = next(self.parameters()).device
        out = []
        for i in range(0, feats.shape[0], batch):
            x = torch.as_tensor(feats[i : i + batch], dtype=torch.float32, device=device)
            out.append(torch.sigmoid(self(x)).cpu().numpy())
        return np.concatenate(out, axis=0)
