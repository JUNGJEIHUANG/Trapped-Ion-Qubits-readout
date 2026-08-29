"""Photon-count distributions for the weakly anchored Bayesian readout.

Zhou et al. (PRL 137, 013601, 2026) model both the dark-state ``g(n)`` and the
bright-state ``f(n)`` as *super-Poissonian* distributions with two parameters
``theta = {alpha, beta}``, deferring the functional form to their Supplemental
Material, which is not part of the article PDF.

ADAPTATION (documented, and the single most important assumption in this
reproduction): we use the Gamma-Poisson mixture, i.e. the negative binomial,

    n ~ Poisson(lambda),  lambda ~ Gamma(shape=alpha, scale=beta)

    mean = alpha * beta
    var  = alpha * beta * (1 + beta)   >= mean  for beta > 0

This is the standard super-Poissonian model for EMCCD photon counting: it has
exactly two parameters, reduces to Poisson as ``beta -> 0`` with ``alpha*beta``
fixed, and admits closed-form weighted moment estimation, which is what the
M-step of the paper's EM iteration requires. If the Supplemental Material
specifies a different two-parameter super-Poissonian, only this module needs to
change -- ``bayes_em`` and ``pi_network`` consume it through the interface below.

The interface any replacement must satisfy is: ``pmf(n, theta)``,
``fit_weighted(n, w)``, ``moments(theta)``, ``sample(theta, size, rng)``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import gammaln

__all__ = ["SuperPoissonian", "Theta", "histogram_overlap"]


@dataclass(frozen=True)
class Theta:
    """Two-parameter super-Poissonian parameters."""

    alpha: float
    beta: float

    def as_array(self) -> np.ndarray:
        return np.array([self.alpha, self.beta], dtype=float)


class SuperPoissonian:
    """Gamma-Poisson (negative binomial) photon-count model."""

    # Guard rails keep the EM iteration inside a numerically sane region.
    MIN_ALPHA, MAX_ALPHA = 1e-3, 1e4
    MIN_BETA, MAX_BETA = 1e-4, 1e3

    @staticmethod
    def pmf(n: np.ndarray, theta: Theta) -> np.ndarray:
        """P(n | alpha, beta), evaluated in log space for stability."""
        n = np.asarray(n, dtype=float)
        a = float(np.clip(theta.alpha, SuperPoissonian.MIN_ALPHA, SuperPoissonian.MAX_ALPHA))
        b = float(np.clip(theta.beta, SuperPoissonian.MIN_BETA, SuperPoissonian.MAX_BETA))
        p = b / (1.0 + b)  # success probability of the NB parameterisation
        log_pmf = (
            gammaln(n + a)
            - gammaln(a)
            - gammaln(n + 1.0)
            + a * np.log1p(-p)
            + n * np.log(p)
        )
        return np.exp(log_pmf)

    @staticmethod
    def moments(theta: Theta) -> tuple:
        a, b = theta.alpha, theta.beta
        return a * b, a * b * (1.0 + b)

    @staticmethod
    def fit_moments(n: np.ndarray, weights: np.ndarray | None = None) -> Theta:
        """Weighted method-of-moments fit -- the M-step of the paper's EM.

        Inverting ``mean = a*b`` and ``var = a*b*(1+b)`` gives
        ``b = var/mean - 1`` and ``a = mean/b``. When the weighted sample is
        under-dispersed (``var <= mean``, possible at low photon counts) the
        estimator is clipped to the Poisson limit rather than allowed to go
        negative.
        """
        n = np.asarray(n, dtype=float)
        w = np.ones_like(n) if weights is None else np.asarray(weights, dtype=float)
        wsum = float(np.sum(w))
        if wsum <= 0:
            return Theta(1.0, 1.0)

        mean = float(np.sum(w * n) / wsum)
        var = float(np.sum(w * (n - mean) ** 2) / wsum)
        mean = max(mean, 1e-6)
        beta = max(var / mean - 1.0, SuperPoissonian.MIN_BETA)
        alpha = max(mean / beta, SuperPoissonian.MIN_ALPHA)
        return Theta(
            float(np.clip(alpha, SuperPoissonian.MIN_ALPHA, SuperPoissonian.MAX_ALPHA)),
            float(np.clip(beta, SuperPoissonian.MIN_BETA, SuperPoissonian.MAX_BETA)),
        )

    @staticmethod
    def sample(theta: Theta, size: int, rng: np.random.Generator) -> np.ndarray:
        lam = rng.gamma(shape=theta.alpha, scale=theta.beta, size=size)
        return rng.poisson(lam)


def histogram_overlap(theta_f: Theta, theta_g: Theta, n_max: int = 2000) -> float:
    """Bhattacharyya overlap ``O(f,g) = sum_n sqrt(f(n) g(n))``.

    Zhou et al. use this to quantify how far the bright and dark distributions
    have merged; 1 means identical, 0 means perfectly separable. Their reported
    regimes are O ~ 0.61 (12.5 ms) and 0.72 (6 ms).
    """
    n = np.arange(0, n_max + 1)
    return float(np.sum(np.sqrt(SuperPoissonian.pmf(n, theta_f) * SuperPoissonian.pmf(n, theta_g))))
