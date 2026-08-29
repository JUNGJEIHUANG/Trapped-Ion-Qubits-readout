"""Reproduction of Kent et al., Phys. Rev. Applied 25, 044048 (2026).

"Efficient measurement of neutral-atom qubits with matched filters".

No source code was released with the paper; this module implements the method
from the text, Sec. III and Appendix A. Every algorithmic choice traceable to
the paper is annotated with its section; every choice the paper leaves open is
marked ADAPTATION and justified.

Method summary
--------------
Per site, a *linear* filter is applied to the pixels in a square boundary of
width ``s`` around the calibrated centre, plus a constant feature:

    y_hat = W . x,   x = [flattened s x s patch, ..., c]

MF-site  (Sec. III A) : x = [patch, c]
MF-array (Sec. III B) : x = [patch, mean of each *other* site's boundary, c]

Weights come from Tikhonov-regularised least squares in closed form
(Eq. A1), ``W = Y X^T (X X^T + alpha I)^-1``, and the boundary size ``s`` and
decision threshold are chosen jointly on the validation split to maximise
classification fidelity (Appendix A 4).

The point of the model is that it is linear, has a closed-form fit, and costs
orders of magnitude less than a CNN -- so the reproduction must not quietly
"improve" it into something nonlinear.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional, Sequence

import numpy as np

from ..common.metrics import classification_fidelity

__all__ = ["MatchedFilterConfig", "MatchedFilterReadout", "SquareFilter", "GaussianFilter"]

NeighborMode = Literal["none", "all", "knn"]


@dataclass
class MatchedFilterConfig:
    """Configuration for the matched-filter readout.

    Parameters
    ----------
    boundary_sizes
        Candidate square boundary widths ``s``. The paper sweeps 2..14 in steps
        of 1 (Appendix A 3).
    thresholds
        Candidate decision thresholds, 0.01..0.99 in steps of 0.01
        (Appendix A 4).
    alpha
        Tikhonov regularisation. The paper sets ``alpha = 0`` for its 28x28
        image with 9 qubits (Appendix A 3). ADAPTATION: the default here is a
        small positive value because a 300-site register with s up to 14 gives
        a 197-dimensional feature vector per site and ``X X^T`` becomes
        ill-conditioned far more often than in the 9-qubit case. Set to 0.0 to
        match the paper exactly; ``alpha`` is selected on validation when
        ``alphas`` holds more than one value.
    alphas
        Candidate ``alpha`` values for validation selection.
    constant_feature
        The augmenting constant ``c`` (Sec. III A).
    neighbor_mode
        ``"none"`` -> MF-site. ``"all"`` -> MF-array using every other site,
        which is what the paper does for its 3x3 array. ``"knn"`` -> MF-array
        restricted to the ``k`` nearest sites; the paper explicitly sanctions
        this for scaling ("limiting the feature vector to include only nearest
        neighbors also results in linear scaling", Sec. IV C).
    n_neighbors
        ``k`` for ``neighbor_mode="knn"``.
    """

    boundary_sizes: Sequence[int] = tuple(range(2, 15))
    thresholds: Sequence[float] = field(
        default_factory=lambda: tuple(np.round(np.arange(0.01, 1.00, 0.01), 2))
    )
    alpha: float = 1e-6
    alphas: Optional[Sequence[float]] = None
    constant_feature: float = 1.0
    neighbor_mode: NeighborMode = "none"
    n_neighbors: int = 8


class _PatchExtractor:
    """Caches per-site square boundaries for every candidate boundary size."""

    def __init__(self, centers: np.ndarray, shape: tuple, boundary_sizes: Sequence[int]):
        self.centers = np.asarray(centers, dtype=float)
        self.H, self.W = shape
        self.n_sites = self.centers.shape[0]
        self._slices: dict[int, list[tuple]] = {}
        for s in boundary_sizes:
            self._slices[s] = [self._box(c, s) for c in self.centers]

    def _box(self, center: np.ndarray, s: int) -> tuple:
        """Square boundary of width ``s`` centred on ``center``.

        The box is clipped to the sensor and then shifted back inside if the
        clip would shorten it, so every site yields a feature vector of the
        same length ``s * s`` -- required for a single shared weight vector.
        """
        r0, c0 = center
        half = s / 2.0
        r_lo = int(np.floor(r0 - half + 0.5))
        c_lo = int(np.floor(c0 - half + 0.5))
        r_lo = int(np.clip(r_lo, 0, max(self.H - s, 0)))
        c_lo = int(np.clip(c_lo, 0, max(self.W - s, 0)))
        return (slice(r_lo, r_lo + s), slice(c_lo, c_lo + s))

    def patches(self, frames: np.ndarray, s: int, site: int) -> np.ndarray:
        """(n_frames, s*s) flattened patch for one site."""
        rs, cs = self._slices[s][site]
        return frames[:, rs, cs].reshape(frames.shape[0], -1)

    def means(self, frames: np.ndarray, s: int) -> np.ndarray:
        """(n_frames, n_sites) mean intensity inside every site's boundary."""
        out = np.empty((frames.shape[0], self.n_sites), dtype=np.float64)
        for k in range(self.n_sites):
            rs, cs = self._slices[s][k]
            out[:, k] = frames[:, rs, cs].reshape(frames.shape[0], -1).mean(axis=1)
        return out


def _ridge_closed_form(X: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
    """Kent et al. Eq. (A1): ``W = Y X^T (X X^T + alpha I)^-1``.

    ``X`` is ``(d, M)`` as in the paper (features x images) and ``y`` is
    ``(M,)``. Solved with ``lstsq`` when ``alpha == 0`` so the exactly-singular
    case degrades to a minimum-norm solution instead of raising.
    """
    XXt = X @ X.T
    Xy = X @ y
    if alpha > 0:
        XXt = XXt + alpha * np.eye(XXt.shape[0])
        try:
            return np.linalg.solve(XXt, Xy)
        except np.linalg.LinAlgError:
            pass
    return np.linalg.lstsq(XXt, Xy, rcond=None)[0]


class MatchedFilterReadout:
    """MF-site / MF-array readout over a calibrated register.

    One independent filter is fitted per site, exactly as in the paper
    ("We evaluate Eq. (A1) for each qubit", Appendix A 3). The boundary size,
    threshold and (optionally) ``alpha`` are also selected per site on the
    validation split.
    """

    def __init__(self, centers: np.ndarray, shape: tuple, config: MatchedFilterConfig | None = None):
        self.config = config or MatchedFilterConfig()
        self.centers = np.asarray(centers, dtype=float)
        self.shape = shape
        self.n_sites = self.centers.shape[0]
        self._ex = _PatchExtractor(self.centers, shape, self.config.boundary_sizes)
        self._neighbors = self._build_neighbors()

        self.weights_: list[np.ndarray] = []
        self.boundary_: np.ndarray = np.zeros(self.n_sites, dtype=int)
        self.threshold_: np.ndarray = np.zeros(self.n_sites, dtype=float)
        self.alpha_: np.ndarray = np.zeros(self.n_sites, dtype=float)

    def _build_neighbors(self) -> Optional[list[np.ndarray]]:
        mode = self.config.neighbor_mode
        if mode == "none":
            return None
        if mode == "all":
            return [np.array([j for j in range(self.n_sites) if j != k]) for k in range(self.n_sites)]
        if mode == "knn":
            d = np.linalg.norm(self.centers[:, None, :] - self.centers[None, :, :], axis=-1)
            np.fill_diagonal(d, np.inf)
            k = min(self.config.n_neighbors, self.n_sites - 1)
            return [np.argsort(d[i])[:k] for i in range(self.n_sites)]
        raise ValueError(f"unknown neighbor_mode {mode!r}")

    def _features(self, frames: np.ndarray, s: int, site: int, means: np.ndarray | None) -> np.ndarray:
        """Feature matrix ``(d, M)`` for one site, matching Fig. 1 of the paper."""
        parts = [self._ex.patches(frames, s, site)]
        if self._neighbors is not None:
            assert means is not None
            parts.append(means[:, self._neighbors[site]])
        parts.append(np.full((frames.shape[0], 1), self.config.constant_feature))
        return np.concatenate(parts, axis=1).T

    def fit(
        self,
        train_frames: np.ndarray,
        train_labels: np.ndarray,
        val_frames: np.ndarray,
        val_labels: np.ndarray,
        verbose: bool = False,
    ) -> "MatchedFilterReadout":
        """Fit weights on train, select (s, threshold, alpha) on validation.

        Training and selection are strictly separated: weights never see the
        validation labels, and the metaparameter search never sees the test
        split. This mirrors Appendix A 3-A 5.
        """
        cfg = self.config
        alphas = list(cfg.alphas) if cfg.alphas is not None else [cfg.alpha]

        train_means = {s: self._ex.means(train_frames, s) for s in cfg.boundary_sizes} if self._neighbors else {}
        val_means = {s: self._ex.means(val_frames, s) for s in cfg.boundary_sizes} if self._neighbors else {}

        self.weights_ = [None] * self.n_sites
        for k in range(self.n_sites):
            y_tr = train_labels[:, k].astype(float)
            y_va = val_labels[:, k].astype(bool)

            best = (-np.inf, None, None, None, None)  # fidelity, W, s, thr, alpha
            for s in cfg.boundary_sizes:
                X_tr = self._features(train_frames, s, k, train_means.get(s))
                X_va = self._features(val_frames, s, k, val_means.get(s))
                for a in alphas:
                    W = _ridge_closed_form(X_tr, y_tr, a)
                    scores = W @ X_va
                    for thr in cfg.thresholds:
                        fid = classification_fidelity(y_va, scores >= thr)
                        if fid > best[0]:
                            best = (fid, W, s, float(thr), float(a))

            _, W, s, thr, a = best
            self.weights_[k] = W
            self.boundary_[k] = s
            self.threshold_[k] = thr
            self.alpha_[k] = a
            if verbose and (k % 50 == 0 or k == self.n_sites - 1):
                print(f"  site {k:4d}/{self.n_sites}: s={s} thr={thr:.2f} val_fidelity={best[0]:.5f}")
        return self

    def decision_scores(self, frames: np.ndarray) -> np.ndarray:
        """Continuous ``y_hat`` per (frame, site). Close to 0 dark, 1 bright."""
        if not self.weights_ or self.weights_[0] is None:
            raise RuntimeError("call fit() before decision_scores()")
        needed = sorted({int(s) for s in self.boundary_})
        means = {s: self._ex.means(frames, s) for s in needed} if self._neighbors else {}

        out = np.empty((frames.shape[0], self.n_sites), dtype=float)
        for k in range(self.n_sites):
            s = int(self.boundary_[k])
            out[:, k] = self.weights_[k] @ self._features(frames, s, k, means.get(s))
        return out

    def predict(self, frames: np.ndarray) -> np.ndarray:
        return self.decision_scores(frames) >= self.threshold_[None, :]

    def predict_proba(self, frames: np.ndarray) -> np.ndarray:
        """Probability-like scores for AUROC/ECE.

        ADAPTATION. The paper thresholds the raw linear score and never needs a
        probability. Site-AIT's Table 1 reports AUROC and ECE, so a calibrated
        score is required. The map used is a per-site logistic recentred on the
        selected threshold; it is monotone, so it leaves accuracy, precision,
        recall, F1 and AUROC untouched and only affects ECE. Its temperature is
        fitted on the validation split by :meth:`calibrate`, so an uncalibrated
        model is never silently reported as calibrated.
        """
        scores = self.decision_scores(frames)
        temp = getattr(self, "temperature_", None)
        if temp is None:
            raise RuntimeError("call calibrate() on the validation split first")
        z = np.clip((scores - self.threshold_[None, :]) / temp[None, :], -700.0, 700.0)
        return 1.0 / (1.0 + np.exp(-z))

    def calibrate(self, val_frames: np.ndarray, val_labels: np.ndarray, grid: Sequence[float] | None = None):
        """Fit the per-site logistic temperature on validation by NLL."""
        grid = grid if grid is not None else np.geomspace(1e-3, 1.0, 40)
        scores = self.decision_scores(val_frames)
        temps = np.empty(self.n_sites)
        for k in range(self.n_sites):
            z = scores[:, k] - self.threshold_[k]
            y = val_labels[:, k].astype(float)
            best_nll, best_t = np.inf, grid[0]
            for t in grid:
                p = np.clip(1.0 / (1.0 + np.exp(-np.clip(z / t, -700.0, 700.0))), 1e-9, 1 - 1e-9)
                nll = -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
                if nll < best_nll:
                    best_nll, best_t = nll, t
            temps[k] = best_t
        self.temperature_ = temps
        return self

    def complexity(self) -> dict:
        """Parameter and multiplication counts, the paper's headline claim."""
        n_params = int(sum(w.size for w in self.weights_))
        return {
            "trainable_parameters": n_params,
            "multiplications_per_frame": n_params,
            "nonlinear_evaluations_per_frame": 0,
            "mean_boundary_size": float(self.boundary_.mean()),
        }


class SquareFilter:
    """Kent et al. Eq. (1): unweighted sum over a square boundary.

    Unsupervised -- the threshold is the only fitted quantity. Included because
    the paper's infidelity-reduction metric eta is defined relative to the
    Gaussian filter, and both traditional filters are needed to reproduce Fig. 2.
    """

    def __init__(self, centers: np.ndarray, shape: tuple, boundary_sizes: Sequence[int] = tuple(range(2, 15))):
        self.centers = np.asarray(centers, dtype=float)
        self._ex = _PatchExtractor(self.centers, shape, boundary_sizes)
        self.boundary_sizes = boundary_sizes
        self.n_sites = self.centers.shape[0]

    def _weights(self, s: int, site: int) -> np.ndarray:
        return np.ones(s * s)

    def fit(self, val_frames: np.ndarray, val_labels: np.ndarray):
        self.boundary_ = np.zeros(self.n_sites, dtype=int)
        self.threshold_ = np.zeros(self.n_sites, dtype=float)
        for k in range(self.n_sites):
            y = val_labels[:, k].astype(bool)
            best = (-np.inf, None, None)
            for s in self.boundary_sizes:
                scores = self._ex.patches(val_frames, s, k) @ self._weights(s, k)
                # Thresholds swept over the observed score range rather than
                # [0.01, 0.99]: an unweighted photon sum is not on a 0-1 scale.
                for thr in np.quantile(scores, np.linspace(0.01, 0.99, 99)):
                    fid = classification_fidelity(y, scores >= thr)
                    if fid > best[0]:
                        best = (fid, s, float(thr))
            _, self.boundary_[k], self.threshold_[k] = best
        return self

    def decision_scores(self, frames: np.ndarray) -> np.ndarray:
        out = np.empty((frames.shape[0], self.n_sites))
        for k in range(self.n_sites):
            s = int(self.boundary_[k])
            out[:, k] = self._ex.patches(frames, s, k) @ self._weights(s, k)
        return out

    def predict(self, frames: np.ndarray) -> np.ndarray:
        return self.decision_scores(frames) >= self.threshold_[None, :]


class GaussianFilter(SquareFilter):
    """Kent et al. Eq. (2): weights are a 2-D Gaussian centred on the site.

    ``sigma`` is obtained by fitting a circular Gaussian to the averaged
    training frames, as the paper specifies; pass it in from
    :func:`common.dataio.locate_sites` or fit it once beforehand.
    """

    def __init__(self, centers, shape, sigma: float, boundary_sizes=tuple(range(2, 15))):
        super().__init__(centers, shape, boundary_sizes)
        self.sigma = float(sigma)

    def _weights(self, s: int, site: int) -> np.ndarray:
        rs, cs = self._ex._slices[s][site]
        rr, cc = np.mgrid[rs, cs]
        r0, c0 = self.centers[site]
        return np.exp(-((rr - r0) ** 2 + (cc - c0) ** 2) / (2 * self.sigma**2)).ravel()
