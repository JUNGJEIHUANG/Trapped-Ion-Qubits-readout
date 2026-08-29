"""Exhaustive detection-model search grids stated in Supplementary App. F."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import product
from typing import Any


@dataclass(frozen=True)
class DetectionCandidate:
    run_id: str
    family: str
    parameters: dict[str, Any]
    seed: int = 1

    def to_dict(self):
        return asdict(self)


def detr_candidates() -> list[DetectionCandidate]:
    grid = product(
        (5e-5, 1e-4, 3e-4, 1e-3), (128, 256, 512), (2, 4, 6),
        (4, 8), (7, 13, 19),
    )
    out = [
        DetectionCandidate(
            f"detr-{i:03d}", "detr_query",
            dict(lr=lr, token_dim=dim, layers=layers, heads=heads, template=size),
        )
        for i, (lr, dim, layers, heads, size) in enumerate(grid, 1)
    ]
    assert len(out) == 216
    return out


def deim_candidates() -> list[DetectionCandidate]:
    grid = product(
        (1e-4, 2.5e-4, 5e-4, 1e-3), ("S", "M", "L"), (1.0, 1.5, 2.0),
        (1.5, 2.0, 3.0), ("default", "double"),
    )
    out = [
        DetectionCandidate(
            f"deim-{i:03d}", "deim_dfine_mal",
            dict(lr=lr, scale=scale, gamma=gamma, box_sigma=box, matching_cost=cost),
        )
        for i, (lr, scale, gamma, box, cost) in enumerate(grid, 1)
    ]
    assert len(out) == 216
    return out


def siteait_candidates() -> list[DetectionCandidate]:
    grid = product((5e-5, 1e-4, 3e-4, 1e-3), (128, 256, 512))
    out = [
        DetectionCandidate(f"siteait-{i:02d}", "site_ait", dict(lr=lr, token_dim=dim))
        for i, (lr, dim) in enumerate(grid, 1)
    ]
    assert len(out) == 12
    return out


def full_detection_plan() -> list[DetectionCandidate]:
    plan = detr_candidates() + deim_candidates() + siteait_candidates()
    assert len(plan) == 444
    return plan
