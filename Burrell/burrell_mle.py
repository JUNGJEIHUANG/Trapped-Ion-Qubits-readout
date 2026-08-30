"""Literature-faithful Burrell et al. (2010) spatial MLE baseline.

This module implements the method described in Burrell et al., Phys. Rev. A
81, 040302 (2010): empirical per-pixel bright/dark count distributions,
optionally conditioned on the states of selected neighbours, followed by
iterative neighbour-state updates.  It deliberately does *not* implement a
crosstalk matrix, PSF kernel inversion, or deconvolution.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence
import numpy as np


@dataclass
class BurrellModel:
    coords: np.ndarray                 # (K, 2), row/column coordinates
    roi_pixels: np.ndarray             # (K, N, 2), brightness-ordered
    neighbours: np.ndarray             # (K, q), -1 padded
    log_ratio: np.ndarray              # (K, 2**q, N, max_count+1)
    max_count: int
    iterations: int = 8


def _check_images(images: np.ndarray) -> np.ndarray:
    x = np.asarray(images)
    if x.ndim != 3:
        raise ValueError("images must have shape (frames, height, width)")
    if not np.isfinite(x).all():
        raise ValueError("images contain NaN or infinity")
    return x


def _sample(images: np.ndarray, pixels: np.ndarray) -> np.ndarray:
    """Return (frames, N) counts at one site's ROI pixels."""
    return images[:, pixels[:, 0], pixels[:, 1]]


def _pmf(values: np.ndarray, max_count: int, alpha: float) -> np.ndarray:
    values = np.rint(values).astype(np.int64).clip(0, max_count)
    h = np.bincount(values, minlength=max_count + 1).astype(float)
    return (h + alpha) / (h.sum() + alpha * (max_count + 1))


def _nearest_neighbours(coords: np.ndarray, q: int) -> np.ndarray:
    k = len(coords)
    out = np.full((k, q), -1, dtype=np.int64)
    if q == 0:
        return out
    d2 = ((coords[:, None, :] - coords[None, :, :]) ** 2).sum(axis=2)
    d2[np.arange(k), np.arange(k)] = np.inf
    order = np.argsort(d2, axis=1, kind="stable")
    n = min(q, max(0, k - 1))
    out[:, :n] = order[:, :n]
    return out


def fit_burrell(
    bright_images: np.ndarray,
    dark_images: np.ndarray,
    coords: np.ndarray,
    *,
    roi_size: int = 10,
    n_neighbours: int = 2,
    max_count: Optional[int] = None,
    alpha: float = 0.5,
    iterations: int = 8,
) -> BurrellModel:
    """Fit empirical distributions using *calibration* bright/dark frames.

    ``bright_images`` and ``dark_images`` must contain frames with known
    all-bright/all-dark states (or a separately supplied calibration protocol).
    ``coords`` are calibrated site coordinates in image-pixel units.
    """
    b, d = _check_images(bright_images), _check_images(dark_images)
    coords = np.asarray(coords, dtype=float)
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError("coords must have shape (K, 2)")
    k = len(coords)
    if k == 0 or roi_size < 1 or n_neighbours < 0:
        raise ValueError("invalid K, roi_size, or n_neighbours")
    if b.shape[1:] != d.shape[1:]:
        raise ValueError("bright and dark image shapes differ")
    # Brightness ordering follows Burrell: rank pixels by mean bright signal.
    contrast = b.mean(axis=0) - d.mean(axis=0)
    radius = max(1, int(np.ceil(np.sqrt(roi_size))))
    roi = np.empty((k, roi_size, 2), dtype=np.int64)
    h, w = b.shape[1:]
    for i, (r, c) in enumerate(np.rint(coords).astype(int)):
        rr, cc = np.mgrid[max(0, r-radius):min(h, r+radius+1),
                           max(0, c-radius):min(w, c+radius+1)]
        cand = np.stack([rr.ravel(), cc.ravel()], axis=1)
        # Prefer positive-contrast pixels; deterministic ties by row/column.
        vals = contrast[cand[:, 0], cand[:, 1]]
        order = np.lexsort((cand[:, 1], cand[:, 0], -vals))
        if len(order) < roi_size:
            raise ValueError(f"ROI size {roi_size} exceeds available pixels at site {i}")
        roi[i] = cand[order[:roi_size]]
    if max_count is None:
        max_count = int(max(np.rint(b).max(), np.rint(d).max()))
    if max_count < 1:
        raise ValueError("max_count must be positive")
    neigh = _nearest_neighbours(coords, n_neighbours)
    qstates = 2 ** n_neighbours
    lr = np.empty((k, qstates, roi_size, max_count + 1), dtype=np.float32)
    # Calibration frames are all-bright/all-dark, hence only all-one/all-zero
    # neighbour configurations are observed. Other configurations are filled
    # with the conservative pooled distributions and are replaced by the
    # mixed-state calibration path in fit_burrell_mixed when available.
    for i in range(k):
        bv, dv = _sample(b, roi[i]), _sample(d, roi[i])
        for state in range(qstates):
            for j in range(roi_size):
                pb = _pmf(bv[:, j], max_count, alpha)
                pd = _pmf(dv[:, j], max_count, alpha)
                lr[i, state, j] = np.log(pb / pd)
    return BurrellModel(coords, roi, neigh, lr, max_count, iterations)


def predict(model: BurrellModel, images: np.ndarray, *, initial: str = "dark") -> np.ndarray:
    """Predict a (frames, K) binary state array by iterative MLE."""
    x = _check_images(images)
    f, k = x.shape[0], len(model.coords)
    states = np.zeros((f, k), dtype=np.uint8) if initial == "dark" else np.ones((f, k), dtype=np.uint8)
    for _ in range(model.iterations):
        old = states.copy()
        for i in range(k):
            neigh = model.neighbours[i]
            cfg = np.zeros(f, dtype=np.int64)
            for bit, n in enumerate(neigh):
                if n >= 0:
                    cfg |= states[:, n].astype(np.int64) << bit
            vals = _sample(x, model.roi_pixels[i]).clip(0, model.max_count)
            score = np.zeros(f, dtype=float)
            for j in range(vals.shape[1]):
                score += model.log_ratio[i, cfg, j, vals[:, j]]
            states[:, i] = (score >= 0).astype(np.uint8)
        if np.array_equal(states, old):
            break
    return states


def fit_burrell_mixed(
    calibration_images: np.ndarray,
    calibration_states: np.ndarray,
    coords: np.ndarray,
    *,
    roi_size: int = 10,
    n_neighbours: int = 2,
    max_count: Optional[int] = None,
    alpha: float = 0.5,
    iterations: int = 8,
) -> BurrellModel:
    """Fit the full neighbour-conditioned model from labelled calibration shots.

    ``calibration_states[m, i]`` is the known 0/1 state of site ``i`` in
    calibration frame ``m``. This is the required path when mixed neighbour
    configurations are available, as in Burrell's pre/post calibration data.
    The default ``n_neighbours=2`` follows the nearest-neighbour model used by
    Burrell et al.; the implementation selects the two closest calibrated
    sites for each target site.
    """
    x = _check_images(calibration_images)
    y = np.asarray(calibration_states, dtype=np.uint8)
    if y.shape != (x.shape[0], len(coords)) or not np.isin(y, [0, 1]).all():
        raise ValueError("calibration_states must have shape (frames, K) and contain 0/1")
    # Reuse ROI and neighbour construction by fitting pooled distributions.
    pooled_b = x[y[:, 0] == 1]
    pooled_d = x[y[:, 0] == 0]
    if len(pooled_b) == 0 or len(pooled_d) == 0:
        raise ValueError("calibration must contain both bright and dark examples")
    base = fit_burrell(pooled_b, pooled_d, coords, roi_size=roi_size,
                       n_neighbours=n_neighbours, max_count=max_count,
                       alpha=alpha, iterations=iterations)
    k, q, n = len(coords), n_neighbours, roi_size
    max_count = base.max_count
    lr = np.empty_like(base.log_ratio)
    for i in range(k):
        neigh = base.neighbours[i]
        cfg = np.zeros(len(y), dtype=np.int64)
        for bit, j in enumerate(neigh):
            if j >= 0:
                cfg |= y[:, j].astype(np.int64) << bit
        for c in range(2 ** q):
            for j in range(n):
                sb = (y[:, i] == 1) & (cfg == c)
                sd = (y[:, i] == 0) & (cfg == c)
                if not sb.any(): sb = y[:, i] == 1
                if not sd.any(): sd = y[:, i] == 0
                pb = _pmf(_sample(x[sb], base.roi_pixels[i])[:, j], max_count, alpha)
                pd = _pmf(_sample(x[sd], base.roi_pixels[i])[:, j], max_count, alpha)
                lr[i, c, j] = np.log(pb / pd)
    return BurrellModel(base.coords, base.roi_pixels, base.neighbours,
                        lr, max_count, iterations)


def accuracy(pred: np.ndarray, truth: np.ndarray) -> float:
    pred, truth = np.asarray(pred), np.asarray(truth)
    if pred.shape != truth.shape:
        raise ValueError("pred and truth shapes differ")
    return float((pred == truth).mean())
