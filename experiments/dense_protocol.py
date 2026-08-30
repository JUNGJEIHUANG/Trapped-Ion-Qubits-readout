"""Exact dense-baseline search space stated in Supplementary App. F."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from itertools import product
from typing import Any, Iterable


MODELS = ("unet", "transunet", "setr", "segmenter", "segman_t")
LEARNING_RATES = (5e-5, 1e-4, 3e-4, 1e-3)
OBJECTIVES = ("bce_dice", "bce", "weighted_bce_dice")
FINAL_SEEDS = (1, 2, 3)


@dataclass(frozen=True)
class Candidate:
    run_id: str
    stage: int
    model: str
    seed: int
    learning_rate: float
    objective: str
    architecture: dict[str, Any] = field(default_factory=dict)
    inherits: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def stage1_candidates() -> list[Candidate]:
    specs: dict[str, Iterable[dict[str, Any]]] = {
        "unet": ({"base_channels": c} for c in (32, 64, 96)),
        "transunet": ({"base_channels": c, "vit_depth": 6} for c in (32, 64)),
        "setr": (
            {"patch_size": p, "decoder": d}
            for p, d in product((16, 8), ("pup", "naive", "mla"))
        ),
        "segmenter": (
            {"patch_size": p, "encoder_depth": d}
            for p, d in product((16, 8, 4), (12, 6))
        ),
        "segman_t": [
            {"output_stride": 32, "imagenet_pretrained": True},
            {"output_stride": 16, "imagenet_pretrained": True},
            {"output_stride": 8, "imagenet_pretrained": True},
            {"output_stride": 8, "imagenet_pretrained": False},
        ],
    }
    out: list[Candidate] = []
    for model in MODELS:
        for index, architecture in enumerate(specs[model], start=1):
            out.append(
                Candidate(
                    run_id=f"s1-{model}-{index:02d}", stage=1, model=model, seed=1,
                    learning_rate=1e-4, objective="bce_dice",
                    architecture=dict(architecture),
                )
            )
    assert len(out) == 21
    return out


def stage2_candidates(
    stage1_winners: dict[str, dict[str, Any]] | None = None,
) -> list[Candidate]:
    """Return 60 runs; unresolved dry plans retain an explicit dependency."""
    out: list[Candidate] = []
    for model in MODELS:
        architecture = dict((stage1_winners or {}).get(model, {}))
        for index, (lr, objective) in enumerate(
            product(LEARNING_RATES, OBJECTIVES), start=1
        ):
            out.append(
                Candidate(
                    run_id=f"s2-{model}-{index:02d}", stage=2, model=model, seed=1,
                    learning_rate=lr, objective=objective, architecture=architecture,
                    inherits=f"stage1:{model}",
                )
            )
    assert len(out) == 60
    return out


def stage3_candidates(
    stage2_winners: dict[str, dict[str, Any]] | None = None,
) -> list[Candidate]:
    out: list[Candidate] = []
    for model in MODELS:
        winner = dict((stage2_winners or {}).get(model, {}))
        architecture = dict(winner.get("architecture", {}))
        lr = float(winner.get("learning_rate", 1e-4))
        objective = str(winner.get("objective", "bce_dice"))
        for seed in FINAL_SEEDS:
            out.append(
                Candidate(
                    run_id=f"s3-{model}-seed{seed}", stage=3, model=model, seed=seed,
                    learning_rate=lr, objective=objective, architecture=architecture,
                    inherits=f"stage2:{model}",
                )
            )
    assert len(out) == 15
    return out


def full_plan() -> list[Candidate]:
    plan = stage1_candidates() + stage2_candidates() + stage3_candidates()
    assert len(plan) == 96
    return plan
