"""Ion--Pixel Mapping (IPM) baseline described in Supplementary App. F."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class IPMConfig:
    contrast_fraction: float = 0.5
    max_pixels: int = 90
    growth_iterations: int = 4


class IonPixelMappingReadout:
    """Calibrated pixel-set growth followed by frozen intensity thresholds.

    Local-peak search is bounded by half the nearest calibrated-site distance,
    preventing a seed from crossing into a neighboring site's Voronoi region.
    This bound is derived from the calibrated lattice rather than tuned on test
    data.
    """

    def __init__(self, nominal_xy: np.ndarray, config: IPMConfig = IPMConfig()):
        self.nominal_xy = np.asarray(nominal_xy, dtype=float)
        if self.nominal_xy.ndim != 2 or self.nominal_xy.shape[1] != 2:
            raise ValueError("nominal_xy must have shape (K,2) in (x,y) order")
        self.config = config
        self.pixel_sets: list[np.ndarray] = []
        self.thresholds: np.ndarray | None = None

    def _local_peak(self, contrast: np.ndarray, site: int) -> tuple[int, int]:
        distances = np.linalg.norm(
            self.nominal_xy[site] - np.delete(self.nominal_xy, site, axis=0), axis=1
        )
        radius = max(1, int(np.floor(distances.min() / 2.0))) if distances.size else 3
        x0, y0 = self.nominal_xy[site]
        h, w = contrast.shape
        x_lo, x_hi = max(0, round(x0) - radius), min(w, round(x0) + radius + 1)
        y_lo, y_hi = max(0, round(y0) - radius), min(h, round(y0) + radius + 1)
        patch = contrast[y_lo:y_hi, x_lo:x_hi]
        dy, dx = np.unravel_index(np.argmax(patch), patch.shape)
        return y_lo + int(dy), x_lo + int(dx)

    @staticmethod
    def _boundary(pixels: set[tuple[int, int]], height: int, width: int):
        out: set[tuple[int, int]] = set()
        for y, x in pixels:
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    q = (y + dy, x + dx)
                    if (dy or dx) and 0 <= q[0] < height and 0 <= q[1] < width:
                        if q not in pixels:
                            out.add(q)
        return out

    def _grow(self, contrast: np.ndarray, peak: tuple[int, int]) -> np.ndarray:
        pixels = {peak}
        for _ in range(self.config.growth_iterations):
            if len(pixels) >= self.config.max_pixels:
                break
            current_mean = float(np.mean([contrast[y, x] for y, x in pixels]))
            cutoff = self.config.contrast_fraction * current_mean
            candidates = [
                q for q in self._boundary(pixels, *contrast.shape) if contrast[q] > cutoff
            ]
            candidates.sort(key=lambda q: float(contrast[q]), reverse=True)
            room = self.config.max_pixels - len(pixels)
            pixels.update(candidates[:room])
        return np.asarray(sorted(pixels), dtype=np.int64)

    @staticmethod
    def _choose_threshold(bright: np.ndarray, dark: np.ndarray) -> float:
        if dark.max() < bright.min():
            return float((dark.max() + bright.min()) / 2.0)
        values = np.unique(np.concatenate([bright, dark]))
        candidates = np.concatenate(
            [[values[0] - 1e-9], (values[:-1] + values[1:]) / 2.0, [values[-1] + 1e-9]]
        )
        labels = np.concatenate([np.ones(bright.size, dtype=bool), np.zeros(dark.size, dtype=bool)])
        scores = np.concatenate([bright, dark])
        accuracies = np.asarray([np.mean((scores >= t) == labels) for t in candidates])
        return float(candidates[int(np.argmax(accuracies))])

    def fit(self, bright_frames: np.ndarray, dark_frames: np.ndarray):
        bright = np.asarray(bright_frames, dtype=float)
        dark = np.asarray(dark_frames, dtype=float)
        if bright.ndim != 3 or dark.ndim != 3 or bright.shape[1:] != dark.shape[1:]:
            raise ValueError("bright_frames and dark_frames must be (N,H,W) with one image shape")
        contrast = bright.mean(axis=0) - dark.mean(axis=0)
        self.pixel_sets = [
            self._grow(contrast, self._local_peak(contrast, k))
            for k in range(self.nominal_xy.shape[0])
        ]
        bright_scores = self.decision_function(bright)
        dark_scores = self.decision_function(dark)
        self.thresholds = np.asarray(
            [self._choose_threshold(bright_scores[:, k], dark_scores[:, k]) for k in range(len(self.pixel_sets))]
        )
        return self

    def decision_function(self, frames: np.ndarray) -> np.ndarray:
        if not self.pixel_sets:
            raise RuntimeError("fit must be called before scoring")
        frames = np.asarray(frames, dtype=float)
        return np.stack(
            [frames[:, pixels[:, 0], pixels[:, 1]].sum(axis=1) for pixels in self.pixel_sets],
            axis=1,
        )

    def predict(self, frames: np.ndarray) -> np.ndarray:
        if self.thresholds is None:
            raise RuntimeError("fit must be called before prediction")
        return self.decision_function(frames) >= self.thresholds
