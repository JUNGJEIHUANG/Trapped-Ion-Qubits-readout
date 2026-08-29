"""Weakly anchored Bayesian-EM readout (Zhou et al., PRL 137, 013601, 2026).

Implements the algorithm in the left panel of the paper's Fig. 2:

    (i)   initialise with prior P(l), observations {n_i}, and theta_f^(0)
    (ii)  E-step: posterior P(l | {n_i}) via Eq. (2); take the mean l-bar
    (iii) M-step: posterior weights
              w_i = l-bar f(n_i) / [(1 - l-bar) g(n_i) + l-bar f(n_i)]
          then weighted moment estimation to update theta_f
    (iv)  iterate until convergence

"Weakly anchored" means only the dark-state distribution ``g`` is calibrated in
advance (from background exposures of empty traps); the bright-state ``f`` is
inferred self-consistently. That asymmetry is the paper's contribution and is
preserved here.

A critical point about what this method *is*
--------------------------------------------
The estimand is ``l``, the bright-state occupation probability of a qubit, and
it is inferred from ``N`` repeated shots of ROI-integrated photon counts. It is
an *ensemble* estimator with a continuous output and a calibrated uncertainty
``Delta l`` -- not a per-shot binary classifier. Site-AIT's task is per-shot,
per-site binary classification from a single frame.

Running this estimator at N = 1 and thresholding it would be a strawman. The
fair adaptation is in ``site_adapter.py``; this module implements the paper's
own task faithfully so the reproduction can be validated against the paper's
own numbers first.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .distributions import SuperPoissonian, Theta

__all__ = ["BayesianEMResult", "weakly_anchored_em", "posterior_over_l", "relative_fidelity"]


@dataclass
class BayesianEMResult:
    """Outputs of the EM iteration: converged theta_f, f(n), l-bar and Delta l."""

    l_mean: float
    l_std: float
    theta_f: Theta
    posterior: np.ndarray
    l_grid: np.ndarray
    n_iterations: int
    trajectory: list


def posterior_over_l(
    counts: np.ndarray,
    theta_f: Theta,
    theta_g: Theta,
    l_grid: np.ndarray,
    log_prior: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Zhou et al. Eq. (2), computed on a discrete ``l`` grid.

    ``P(n | l) = (1 - l) g(n) + l f(n)`` is the binary mixture of Eq. (1); the
    product over shots is accumulated in log space, which matters because the
    paper's regime (N up to ~1000 shots) underflows a direct product.
    """
    counts = np.asarray(counts, dtype=float).reshape(-1)
    f = SuperPoissonian.pmf(counts, theta_f)
    g = SuperPoissonian.pmf(counts, theta_g)

    mix = (1.0 - l_grid)[:, None] * g[None, :] + l_grid[:, None] * f[None, :]
    log_like = np.sum(np.log(np.clip(mix, 1e-300, None)), axis=1)

    if log_prior is None:
        log_prior = np.zeros_like(l_grid)  # uniform prior, as in the paper
    log_post = log_like + log_prior
    log_post -= log_post.max()

    post = np.exp(log_post)
    dl = float(l_grid[1] - l_grid[0])
    return post / max(np.sum(post) * dl, 1e-300)


def _posterior_moments(post: np.ndarray, l_grid: np.ndarray) -> tuple:
    dl = float(l_grid[1] - l_grid[0])
    mean = float(np.sum(post * l_grid) * dl)
    var = float(np.sum(post * (l_grid - mean) ** 2) * dl)
    return mean, float(np.sqrt(max(var, 0.0)))


def weakly_anchored_em(
    counts: np.ndarray,
    theta_g: Theta,
    theta_f_init: Optional[Theta] = None,
    n_grid: int = 501,
    max_iter: int = 10,
    tol: float = 1e-5,
) -> BayesianEMResult:
    """Run the weakly anchored Bayesian-EM iteration for one ROI.

    Parameters
    ----------
    counts
        ``(N,)`` ROI-integrated photon counts over ``N`` repeated shots.
    theta_g
        Pre-anchored dark-state parameters, calibrated from background frames.
    theta_f_init
        Initial guess for the bright-state parameters. Defaults to a
        moment-matched fit of all counts, which is deliberately crude -- the
        paper's point is that EM recovers ``f`` without a good anchor.
    max_iter
        The paper fixes 10 iterations per task for its timing comparison and
        shows ``alpha_f, beta_f`` converging within 10 iterations (Fig. 3d).
    """
    counts = np.asarray(counts, dtype=float).reshape(-1)
    l_grid = np.linspace(0.0, 1.0, n_grid)
    theta_f = theta_f_init or SuperPoissonian.fit_moments(counts)

    trajectory, post, l_mean, l_std = [], None, 0.5, 0.0
    for it in range(max_iter):
        # E-step
        post = posterior_over_l(counts, theta_f, theta_g, l_grid)
        l_mean, l_std = _posterior_moments(post, l_grid)

        # M-step
        f = SuperPoissonian.pmf(counts, theta_f)
        g = SuperPoissonian.pmf(counts, theta_g)
        denom = np.clip((1.0 - l_mean) * g + l_mean * f, 1e-300, None)
        w = l_mean * f / denom
        theta_new = SuperPoissonian.fit_moments(counts, w)

        trajectory.append((it, l_mean, l_std, theta_new.alpha, theta_new.beta))
        delta = abs(theta_new.alpha - theta_f.alpha) + abs(theta_new.beta - theta_f.beta)
        theta_f = theta_new
        if delta < tol:
            break

    post = posterior_over_l(counts, theta_f, theta_g, l_grid)
    l_mean, l_std = _posterior_moments(post, l_grid)
    return BayesianEMResult(l_mean, l_std, theta_f, post, l_grid, len(trajectory), trajectory)


def relative_fidelity(l_reference: float, l_inferred: float) -> float:
    """Zhou et al. Eq. (3): squared overlap between two equal-phase states.

        F = [sqrt(l_th * l_bar) + sqrt((1 - l_th)(1 - l_bar))]^2

    The exponent 2 is on the bracket, not a denominator -- verified against the
    typeset equation on p. 3 of the article. (Plain-text extraction of the PDF
    renders the exponent on its own line, where it reads like a fraction bar;
    that reading is also ruled out on physical grounds, since it would give
    F = 0.5 for a perfect match and could never produce the paper's reported
    F >= 99.99%.)

    With l_th == l_bar this is exactly 1, as a state overlap must be.
    """
    a = float(np.clip(l_reference, 0.0, 1.0))
    b = float(np.clip(l_inferred, 0.0, 1.0))
    return float((np.sqrt(a * b) + np.sqrt((1 - a) * (1 - b))) ** 2)
