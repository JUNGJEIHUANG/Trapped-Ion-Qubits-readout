"""Readout metrics shared by both reproduced baselines.

Two metric families are kept deliberately separate:

*   The Site-AIT family (per-ion accuracy, bright-class precision/recall/F1,
    AUROC, 15-bin ECE) so that a reproduced baseline can be dropped straight
    into Table 1 of the manuscript.
*   The source-paper family (classification fidelity F, cross-fidelity, and
    infidelity reduction eta) so that a reproduction can first be validated
    against the numbers printed in the paper it comes from.

Reporting both is what makes the comparison auditable: the second family
proves the reproduction is faithful, the first family makes it comparable.

References
----------
Kent et al., Phys. Rev. Applied 25, 044048 (2026), Eqs. (3)-(5).
Site-AIT Supplementary, app:metric_definitions.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "per_ion_accuracy",
    "bright_class_prf",
    "auroc",
    "expected_calibration_error",
    "classification_fidelity",
    "cross_fidelity",
    "infidelity_reduction",
    "localization_dispersion",
    "summarize",
]


def _as_flat_bool(a) -> np.ndarray:
    return np.asarray(a).reshape(-1).astype(bool)


def per_ion_accuracy(y_true, y_pred, valid=None) -> float:
    """Pooled per-ion accuracy over all (frame, site) decisions."""
    y_true = _as_flat_bool(y_true)
    y_pred = _as_flat_bool(y_pred)
    if valid is None:
        return float(np.mean(y_true == y_pred))
    v = _as_flat_bool(valid)
    return float(np.sum((y_true == y_pred) & v) / max(np.sum(v), 1))


def bright_class_prf(y_true, y_pred, valid=None):
    """Bright-class precision, recall and F1, pooled over all site decisions.

    Pooled rather than macro-averaged over frames, matching
    ``app:metric_definitions`` in the Site-AIT supplementary material.
    """
    y_true = _as_flat_bool(y_true)
    y_pred = _as_flat_bool(y_pred)
    v = np.ones_like(y_true) if valid is None else _as_flat_bool(valid)

    tp = float(np.sum(v & y_true & y_pred))
    fp = float(np.sum(v & ~y_true & y_pred))
    fn = float(np.sum(v & y_true & ~y_pred))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def auroc(y_true, scores) -> float:
    """Rank-based AUROC; ties receive averaged ranks."""
    y_true = _as_flat_bool(y_true)
    scores = np.asarray(scores).reshape(-1).astype(float)
    n_pos = int(np.sum(y_true))
    n_neg = int(y_true.size - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(scores.size, dtype=float)
    sorted_scores = scores[order]
    i = 0
    while i < sorted_scores.size:
        j = i
        while j + 1 < sorted_scores.size and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        ranks[order[i : j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return float((np.sum(ranks[y_true]) - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def expected_calibration_error(y_true, probs, n_bins: int = 15) -> float:
    """Equal-width binned ECE (15 bins, matching the manuscript)."""
    y_true = _as_flat_bool(y_true)
    probs = np.asarray(probs).reshape(-1).astype(float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(probs, edges[1:-1], right=False), 0, n_bins - 1)

    ece = 0.0
    for b in range(n_bins):
        m = idx == b
        if not np.any(m):
            continue
        ece += (np.sum(m) / probs.size) * abs(np.mean(y_true[m]) - np.mean(probs[m]))
    return float(ece)


def classification_fidelity(y_true, y_pred, valid=None) -> float:
    """Kent et al. Eq. (3): F = 1 - [P(B_pred|D) + P(D_pred|B)] / 2.

    This is a *balanced* metric: it weights the two error directions equally
    regardless of class prior, so it does not coincide with accuracy unless
    the classes are balanced.
    """
    y_true = _as_flat_bool(y_true)
    y_pred = _as_flat_bool(y_pred)
    v = np.ones_like(y_true) if valid is None else _as_flat_bool(valid)

    dark = v & ~y_true
    bright = v & y_true
    p_bright_given_dark = float(np.mean(y_pred[dark])) if np.any(dark) else 0.0
    p_dark_given_bright = float(np.mean(~y_pred[bright])) if np.any(bright) else 0.0
    return 1.0 - 0.5 * (p_bright_given_dark + p_dark_given_bright)


def cross_fidelity(y_pred_sites: np.ndarray, k: int, l: int) -> float:
    """Kent et al. Eq. (4): cross-fidelity between predicted states at two sites.

    ``y_pred_sites`` has shape ``(n_frames, n_sites)`` and holds *predicted*
    states -- the metric measures predicted correlation between neighbours,
    which is what exposes crosstalk leakage, so it never touches the labels.
    """
    pred = np.asarray(y_pred_sites).astype(bool)
    pk, pl = pred[:, k], pred[:, l]

    p_dk_given_bl = float(np.mean(~pk[pl])) if np.any(pl) else 0.0
    p_bk_given_dl = float(np.mean(pk[~pl])) if np.any(~pl) else 0.0
    return 1.0 - (p_dk_given_bl + p_bk_given_dl)


def infidelity_reduction(fidelity_reference: float, fidelity_model: float) -> float:
    """Kent et al. Eq. (5): fractional drop in infidelity vs a reference model."""
    denom = 1.0 - fidelity_reference
    if denom <= 0:
        return float("nan")
    return float((denom - (1.0 - fidelity_model)) / denom)


def localization_dispersion(residuals, valid=None) -> float:
    """Mean-subtracted RMS of the coordinate residual, in camera pixels.

    Mirrors ``app:metric_definitions``: the valid-site mean residual is removed
    first, so this is a precision (reproducibility) measure, not an accuracy
    measure against an absolute centroid.
    """
    res = np.asarray(residuals, dtype=float).reshape(-1, 2)
    if valid is None:
        sel = np.ones(res.shape[0], dtype=bool)
    else:
        sel = _as_flat_bool(valid)
    r = res[sel]
    if r.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean(np.sum((r - r.mean(axis=0)) ** 2, axis=1))))


def summarize(y_true, y_pred, probs=None, valid=None, n_bins: int = 15) -> dict:
    """One call returning both metric families."""
    precision, recall, f1 = bright_class_prf(y_true, y_pred, valid)
    out = {
        "accuracy": per_ion_accuracy(y_true, y_pred, valid),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "fidelity": classification_fidelity(y_true, y_pred, valid),
    }
    if probs is not None:
        out["auroc"] = auroc(y_true, probs)
        out["ece"] = expected_calibration_error(y_true, probs, n_bins)
    return out
