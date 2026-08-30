"""Validation-only mask-to-site decoder selection for dense baselines."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


RADII = (2, 3, 4, 5)
THRESHOLDS = tuple(float(x) for x in np.round(np.arange(0.30, 0.701, 0.01), 2))


@dataclass(frozen=True)
class DecoderSelection:
    radius: int
    threshold: float
    youden_j: float
    accuracy: float

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


def disk_average_probabilities(
    probability_maps: np.ndarray, site_xy: np.ndarray, radius: int
) -> np.ndarray:
    """Average N probability maps in clipped disks around calibrated (x,y) sites."""
    maps = np.asarray(probability_maps, dtype=np.float64)
    if maps.ndim == 4 and maps.shape[1] == 1:
        maps = maps[:, 0]
    if maps.ndim != 3:
        raise ValueError("probability_maps must have shape (N,H,W) or (N,1,H,W)")
    sites = np.asarray(site_xy, dtype=np.float64)
    if sites.ndim != 2 or sites.shape[1] != 2:
        raise ValueError("site_xy must have shape (K,2) in (x,y) order")

    _, height, width = maps.shape
    scores = np.empty((maps.shape[0], sites.shape[0]), dtype=np.float64)
    for k, (x0, y0) in enumerate(sites):
        x_lo = max(0, int(np.floor(x0 - radius)))
        x_hi = min(width, int(np.ceil(x0 + radius)) + 1)
        y_lo = max(0, int(np.floor(y0 - radius)))
        y_hi = min(height, int(np.ceil(y0 + radius)) + 1)
        yy, xx = np.mgrid[y_lo:y_hi, x_lo:x_hi]
        keep = (xx - x0) ** 2 + (yy - y0) ** 2 <= radius**2
        if not np.any(keep):
            raise ValueError(f"empty disk for site {k}")
        scores[:, k] = maps[:, y_lo:y_hi, x_lo:x_hi][:, keep].mean(axis=1)
    return scores


def states_from_masks(masks: np.ndarray, site_xy: np.ndarray) -> np.ndarray:
    masks = np.asarray(masks)
    if masks.ndim == 4 and masks.shape[1] == 1:
        masks = masks[:, 0]
    if masks.ndim != 3:
        raise ValueError("masks must have shape (N,H,W) or (N,1,H,W)")
    sites = np.asarray(site_xy)
    xs = np.rint(sites[:, 0]).astype(int).clip(0, masks.shape[2] - 1)
    ys = np.rint(sites[:, 1]).astype(int).clip(0, masks.shape[1] - 1)
    return masks[:, ys, xs] >= 0.5


def _youden(labels: np.ndarray, predictions: np.ndarray) -> float:
    labels = labels.astype(bool).ravel()
    predictions = predictions.astype(bool).ravel()
    positives = labels.sum()
    negatives = (~labels).sum()
    tpr = ((predictions & labels).sum() / positives) if positives else 0.0
    fpr = ((predictions & ~labels).sum() / negatives) if negatives else 0.0
    return float(tpr - fpr)


def select_decoder(
    probability_maps: np.ndarray,
    state_labels: np.ndarray,
    site_xy: np.ndarray,
    radii: tuple[int, ...] = RADII,
    thresholds: tuple[float, ...] = THRESHOLDS,
) -> DecoderSelection:
    """Select threshold by Youden J within radius, then radius by accuracy."""
    labels = np.asarray(state_labels, dtype=bool)
    scores_by_radius = {
        int(radius): disk_average_probabilities(probability_maps, site_xy, radius)
        for radius in radii
    }
    return select_decoder_from_scores(scores_by_radius, labels, thresholds)


def select_decoder_from_scores(
    scores_by_radius: dict[int, np.ndarray],
    state_labels: np.ndarray,
    thresholds: tuple[float, ...] = THRESHOLDS,
) -> DecoderSelection:
    """Memory-efficient selection from precomputed per-site validation scores."""
    labels = np.asarray(state_labels, dtype=bool)
    best: DecoderSelection | None = None
    for radius, scores in scores_by_radius.items():
        scores = np.asarray(scores)
        if scores.shape != labels.shape:
            raise ValueError(
                f"radius {radius} scores {scores.shape} do not match labels {labels.shape}"
            )
        radius_best: DecoderSelection | None = None
        for threshold in thresholds:
            pred = scores >= threshold
            selection = DecoderSelection(
                radius=int(radius), threshold=float(threshold),
                youden_j=_youden(labels, pred), accuracy=float(np.mean(pred == labels)),
            )
            key = (selection.youden_j, selection.accuracy, -selection.threshold)
            if radius_best is None or key > (
                radius_best.youden_j, radius_best.accuracy, -radius_best.threshold
            ):
                radius_best = selection
        assert radius_best is not None
        key = (radius_best.accuracy, radius_best.youden_j, -radius_best.radius)
        if best is None or key > (best.accuracy, best.youden_j, -best.radius):
            best = radius_best
    assert best is not None
    return best
