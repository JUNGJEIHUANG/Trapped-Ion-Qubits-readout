"""Plan and execute the manuscript's 96-run dense-baseline protocol.

The default is a safe dry run that writes the complete dependent JSONL plan and
launches no training.  With ``--execute``, stages run sequentially, only
validation results select winners, and completed runs are resumable.

The four in-repository architectures call ``train_main.py``. SegMAN-T must use
the authors' implementation; ``--segman-command`` is a Python format string
that must write ``candidate_result.json`` in ``{output_dir}``.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

from .dense_protocol import Candidate, full_plan, stage1_candidates, stage2_candidates, stage3_candidates


def _objective_args(name: str) -> list[str]:
    if name == "bce_dice":
        return ["--bce_weight", "1", "--dice_weight", "1"]
    if name == "bce":
        return ["--bce_weight", "1", "--dice_weight", "0"]
    if name == "weighted_bce_dice":
        return ["--bce_weight", "1", "--dice_weight", "1", "--auto_pos_weight"]
    raise ValueError(f"unknown objective: {name}")


def _internal_arch_args(candidate: Candidate) -> tuple[str, list[str]]:
    a = candidate.architecture
    if candidate.model == "unet":
        return "standard_unet", ["--dense_base_channels", str(a["base_channels"])]
    if candidate.model == "transunet":
        return "vit_unet", [
            "--dense_base_channels", str(a["base_channels"]),
            "--dense_vit_depth", str(a["vit_depth"]),
        ]
    if candidate.model == "setr":
        return "setr", [
            "--dense_patch_size", str(a["patch_size"]),
            "--dense_setr_decoder", str(a["decoder"]),
        ]
    if candidate.model == "segmenter":
        return "segmenter", [
            "--dense_patch_size", str(a["patch_size"]),
            "--dense_segmenter_depth", str(a["encoder_depth"]),
        ]
    raise ValueError(candidate.model)


def build_command(
    candidate: Candidate,
    output_dir: Path,
    python: str,
    calibration_mask: str,
    segman_command: str | None,
) -> list[str]:
    if candidate.model == "segman_t":
        if not segman_command:
            raise RuntimeError("SegMAN-T execution requires --segman-command")
        values = {
            "output_dir": str(output_dir), "seed": candidate.seed,
            "lr": candidate.learning_rate, "objective": candidate.objective,
            **candidate.architecture,
        }
        return shlex.split(segman_command.format(**values), posix=False)

    model_arch, architecture_args = _internal_arch_args(candidate)
    return [
        python, "train_main.py", "--model_arch", model_arch, "--from_scratch",
        "--sample_sizes", "50000", "--epochs", "100", "--batch_size", "48",
        "--learning_rate", str(candidate.learning_rate),
        "--random_state", str(candidate.seed),
        "--early_stopping_metric", "val_ion_acc",
        "--dense_calibration_mask", calibration_mask,
        "--output_dir", str(output_dir),
        *_objective_args(candidate.objective), *architecture_args,
    ]


def _load_result(output_dir: Path) -> dict[str, Any]:
    path = output_dir / "candidate_result.json"
    if not path.exists():
        raise RuntimeError(f"training command did not produce {path}")
    result = json.loads(path.read_text(encoding="utf-8"))
    for key in ("val_ion_acc", "decoder_radius", "decoder_threshold"):
        if key not in result:
            raise RuntimeError(f"{path} is missing {key}")
    return result


def _run_stage(
    candidates: list[Candidate], args: argparse.Namespace, root: Path
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for candidate in candidates:
        output_dir = root / candidate.run_id
        output_dir.mkdir(parents=True, exist_ok=True)
        command = build_command(
            candidate, output_dir, args.python, args.calibration_mask, args.segman_command
        )
        print(candidate.run_id, subprocess.list2cmdline(command))
        if not args.execute:
            continue
        result_path = output_dir / "candidate_result.json"
        if not result_path.exists() or not args.resume:
            env = dict(os.environ)
            env["IMG_DIR"] = args.data_root
            subprocess.run(command, check=True, cwd=args.repo_root, env=env)
        results.append({"candidate": candidate.to_dict(), "result": _load_result(output_dir)})
    return results


def _winner_payload(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    winners: dict[str, dict[str, Any]] = {}
    for row in rows:
        candidate = row["candidate"]
        model = candidate["model"]
        score = float(row["result"]["val_ion_acc"])
        if model not in winners or score > float(winners[model]["val_ion_acc"]):
            winners[model] = {**candidate, **row["result"]}
    return winners


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", help="launch training")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--repo-root", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--output-root", default="outputs/dense_protocol")
    parser.add_argument("--data-root", default="")
    parser.add_argument("--calibration-mask", default="")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--segman-command", default=None)
    parser.add_argument("--write-plan", default="dense_protocol_plan.jsonl")
    args = parser.parse_args()
    args.repo_root = str(Path(args.repo_root).resolve())
    root = Path(args.repo_root) / args.output_root
    root.mkdir(parents=True, exist_ok=True)

    plan = full_plan()
    plan_path = root / args.write_plan
    plan_path.write_text(
        "".join(json.dumps(c.to_dict(), sort_keys=True) + "\n" for c in plan),
        encoding="utf-8",
    )
    print(f"wrote {len(plan)}-run plan to {plan_path}")
    if not args.execute:
        print("dry run only: no training commands were launched")
        return
    if not args.data_root or not args.calibration_mask:
        parser.error("--execute requires --data-root and --calibration-mask")

    if not args.segman_command:

        parser.error("--execute requires --segman-command for the complete 96-run protocol")

    if not Path(args.data_root).is_dir():

        parser.error(f"data root does not exist: {args.data_root}")

    if not Path(args.calibration_mask).is_file():

        parser.error(f"calibration mask does not exist: {args.calibration_mask}")

    stage1_rows = _run_stage(stage1_candidates(), args, root)
    stage1_winners = _winner_payload(stage1_rows)
    (root / "stage1_winners.json").write_text(
        json.dumps(stage1_winners, indent=2), encoding="utf-8"
    )

    stage2_arch = {m: row["architecture"] for m, row in stage1_winners.items()}
    stage2_rows = _run_stage(stage2_candidates(stage2_arch), args, root)
    stage2_winners = _winner_payload(stage2_rows)
    (root / "stage2_winners.json").write_text(
        json.dumps(stage2_winners, indent=2), encoding="utf-8"
    )

    stage3_payload = {
        m: {
            "architecture": row["architecture"],
            "learning_rate": row["learning_rate"],
            "objective": row["objective"],
        }
        for m, row in stage2_winners.items()
    }
    _run_stage(stage3_candidates(stage3_payload), args, root)


if __name__ == "__main__":
    main()
