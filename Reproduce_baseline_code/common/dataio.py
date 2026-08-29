"""Dataset interface and preprocessing for the reproduced baselines.

The real Site-AIT corpus (52,000 sCMOS frames of a 300-site Yb+ register,
456 x 88 px) is not shipped with this code. Everything downstream of
:class:`ReadoutDataset` is written against that interface only, so plugging in
the real corpus means implementing one loader; nothing else changes.

Two loaders are provided:

*   :func:`load_npz` -- the real path. Expects a ``.npz`` holding the frames,
    labels, lattice and split indices (see the docstring for the exact keys).
*   :func:`synthetic_register` -- a physically-motivated simulator of the same
    register, used by the unit tests and the smoke run so the implementations
    are executable and verifiable before the real data is attached.

Preprocessing follows Kent et al. Appendix A 1: subtract the mean training
intensity, normalise by the global intensity range, and locate site centres by
peak finding on the mean training frame followed by Gaussian fits.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Optional

import numpy as np
import warnings

from scipy.optimize import OptimizeWarning, curve_fit

__all__ = [
    "ReadoutDataset",
    "Splits",
    "load_npz",
    "synthetic_register",
    "preprocess",
    "locate_sites",
]


@dataclass
class Splits:
    """Frame indices for the three partitions. Never overlapping."""

    train: np.ndarray
    val: np.ndarray
    test: np.ndarray

    def __post_init__(self) -> None:
        for a, b, name in (
            (self.train, self.val, "train/val"),
            (self.train, self.test, "train/test"),
            (self.val, self.test, "val/test"),
        ):
            if np.intersect1d(a, b).size:
                raise ValueError(f"{name} split indices overlap")


@dataclass
class ReadoutDataset:
    """A register-readout corpus.

    Attributes
    ----------
    frames : (n_frames, H, W) float32
        Raw camera frames.
    labels : (n_frames, n_sites) bool
        Reference bright/dark state per site. ``True`` == bright.
    lattice : (n_sites, 2) float
        Nominal calibrated site centres ``C^0`` as ``(row, col)``.
    splits : Splits
    valid : (n_frames, n_sites) bool or None
        Site validity mask (e.g. the permanent-dark 174-Yb+ site is excluded).
    """

    frames: np.ndarray
    labels: np.ndarray
    lattice: np.ndarray
    splits: Splits
    valid: Optional[np.ndarray] = None

    @property
    def n_sites(self) -> int:
        return int(self.lattice.shape[0])

    @property
    def shape(self) -> tuple:
        return tuple(self.frames.shape[1:])

    def subset(self, idx: np.ndarray):
        """Frames, labels and validity mask for a set of frame indices."""
        idx = np.asarray(idx)
        v = None if self.valid is None else self.valid[idx]
        return self.frames[idx], self.labels[idx], v


def load_npz(path: str) -> ReadoutDataset:
    """Load the real corpus from a ``.npz`` archive.

    Required keys: ``frames``, ``labels``, ``lattice``, ``train_idx``,
    ``val_idx``, ``test_idx``. Optional key: ``valid``.
    """
    z = np.load(path, allow_pickle=False)
    return ReadoutDataset(
        frames=z["frames"].astype(np.float32),
        labels=z["labels"].astype(bool),
        lattice=z["lattice"].astype(float),
        splits=Splits(train=z["train_idx"], val=z["val_idx"], test=z["test_idx"]),
        valid=z["valid"].astype(bool) if "valid" in z.files else None,
    )


def synthetic_register(
    n_frames: int = 3000,
    grid: tuple = (10, 30),
    spacing: float = 6.0,
    psf_sigma: float = 2.5,
    bright_photons: float = 220.0,
    dark_leak: float = 0.04,
    read_noise: float = 3.0,
    background: float = 12.0,
    drift_px: float = 0.35,
    seed: int = 0,
) -> ReadoutDataset:
    """Simulate a dense register whose spacing-to-PSF-width ratio matches the
    manuscript (default 6.0 / 2.5 = 2.4).

    The point of this simulator is not to stand in for the experiment; it is to
    make the reproduced estimators runnable and testable, and to reproduce the
    one property both source papers are actually about -- neighbouring point
    sources whose PSFs overlap enough to induce readout crosstalk.
    """
    rng = np.random.default_rng(seed)
    n_rows, n_cols = grid
    n_sites = n_rows * n_cols

    margin = 4.0 * psf_sigma
    H = int(np.ceil((n_rows - 1) * spacing + 2 * margin))
    W = int(np.ceil((n_cols - 1) * spacing + 2 * margin))

    rr, cc = np.meshgrid(np.arange(n_rows), np.arange(n_cols), indexing="ij")
    lattice = np.stack([rr.ravel() * spacing + margin, cc.ravel() * spacing + margin], axis=1)

    # Static per-site illumination inhomogeneity, as in real registers.
    site_gain = rng.uniform(0.85, 1.15, size=n_sites)

    ys, xs = np.mgrid[0:H, 0:W]
    frames = np.empty((n_frames, H, W), dtype=np.float32)

    # Frame classes mirroring the manuscript's training composition:
    # all-bright, all-dark, and binomial pi/2-pulse frames.
    kinds = rng.choice([0, 1, 2], size=n_frames, p=[0.2, 0.2, 0.6])
    labels = np.empty((n_frames, n_sites), dtype=bool)
    labels[kinds == 0] = True
    labels[kinds == 1] = False
    n_mixed = int(np.sum(kinds == 2))
    labels[kinds == 2] = rng.random((n_mixed, n_sites)) < 0.5

    for i in range(n_frames):
        shift = rng.normal(0.0, drift_px, size=2)
        rate = np.full((H, W), background, dtype=float)
        amp = np.where(labels[i], 1.0, dark_leak) * site_gain * bright_photons
        for k in range(n_sites):
            r0, c0 = lattice[k] + shift
            # 4-sigma support keeps the simulator O(n_sites) rather than O(H*W*n_sites).
            r_lo, r_hi = int(max(0, r0 - 4 * psf_sigma)), int(min(H, r0 + 4 * psf_sigma + 1))
            c_lo, c_hi = int(max(0, c0 - 4 * psf_sigma)), int(min(W, c0 + 4 * psf_sigma + 1))
            dy = ys[r_lo:r_hi, c_lo:c_hi] - r0
            dx = xs[r_lo:r_hi, c_lo:c_hi] - c0
            rate[r_lo:r_hi, c_lo:c_hi] += (
                amp[k]
                / (2 * np.pi * psf_sigma**2)
                * np.exp(-(dy**2 + dx**2) / (2 * psf_sigma**2))
            )
        frames[i] = rng.poisson(rate) + rng.normal(0.0, read_noise, size=(H, W))

    perm = rng.permutation(n_frames)
    n_tr = int(0.6 * n_frames)
    n_va = int(0.2 * n_frames)
    splits = Splits(
        train=perm[:n_tr], val=perm[n_tr : n_tr + n_va], test=perm[n_tr + n_va :]
    )
    return ReadoutDataset(frames=frames, labels=labels, lattice=lattice, splits=splits)


def preprocess(ds: ReadoutDataset) -> ReadoutDataset:
    """Kent et al. Appendix A 1 preprocessing.

    Statistics come from the training partition alone, then are applied
    unchanged to all three partitions -- validation and test never influence
    the normalisation.
    """
    train = ds.frames[ds.splits.train]
    mean_intensity = float(train.mean())
    centered = ds.frames - mean_intensity
    scale = float(centered[ds.splits.train].max() - centered[ds.splits.train].min())
    if scale <= 0:
        scale = 1.0
    return replace(ds, frames=(centered / scale).astype(np.float32))


def _gaussian_2d(coords, amp, r0, c0, sigma, offset):
    r, c = coords
    return (amp * np.exp(-((r - r0) ** 2 + (c - c0) ** 2) / (2 * sigma**2)) + offset).ravel()


def locate_sites(
    ds: ReadoutDataset, window: int = 5, fallback_to_nominal: bool = True
) -> np.ndarray:
    """Refine site centres by Gaussian fits on the mean training frame.

    Kent et al. seed the fits with ``skimage.feature.peak_local_max``. Here the
    nominal lattice ``C^0`` is already known, so it is used as the seed
    directly -- a strictly better initialisation that keeps site *identity*
    unambiguous, which peak finding on a dense register cannot guarantee.
    """
    mean_frame = ds.frames[ds.splits.train].mean(axis=0)
    H, W = mean_frame.shape
    centers = np.empty_like(ds.lattice)

    for k, (r0, c0) in enumerate(ds.lattice):
        r_lo, r_hi = int(max(0, round(r0) - window)), int(min(H, round(r0) + window + 1))
        c_lo, c_hi = int(max(0, round(c0) - window)), int(min(W, round(c0) + window + 1))
        patch = mean_frame[r_lo:r_hi, c_lo:c_hi]
        rr, cc = np.mgrid[r_lo:r_hi, c_lo:c_hi]
        p0 = (float(patch.max() - patch.min()), r0, c0, 2.0, float(patch.min()))
        try:
            with warnings.catch_warnings():
                # A flat or saturated patch makes the covariance unestimable.
                # The fitted centre is still usable and is range-checked below,
                # so this warning carries no information for the caller.
                warnings.simplefilter("ignore", OptimizeWarning)
                popt, _ = curve_fit(_gaussian_2d, (rr, cc), patch.ravel(), p0=p0, maxfev=5000)
            fit_r, fit_c = popt[1], popt[2]
            # Reject fits that wandered outside the seed window.
            if abs(fit_r - r0) > window or abs(fit_c - c0) > window:
                raise RuntimeError("fit escaped the seed window")
            centers[k] = (fit_r, fit_c)
        except (RuntimeError, ValueError):
            if not fallback_to_nominal:
                raise
            centers[k] = (r0, c0)
    return centers
