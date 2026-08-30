"""Dry-plan or execute the 216/216/12 validation-only detection searches."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

from .detection_protocol import full_detection_plan


def _command(candidate, output, args):
    p = candidate.parameters
    if candidate.family == "deim_dfine_mal":
        if not args.deim_command:
            raise RuntimeError("DEIM execution requires --deim-command")
        return shlex.split(
            args.deim_command.format(output_dir=output, seed=1, **p), posix=False
        )
    common = [
        args.python, "train_main.py", "--sample_sizes", "50000", "--epochs", "100",
        "--batch_size", "48", "--random_state", "1", "--learning_rate", str(p["lr"]),
        "--early_stopping_metric", "val_ion_acc", "--site_dia_label_dir",
        args.site_label_dir, "--pretrained_ckpt", args.pretrained_ckpt,
        "--load_mode", "backbone", "--output_dir", str(output),
    ]
    if candidate.family == "detr_query":
        return common + [
            "--model_arch", "detr_query", "--detr_token_dim", str(p["token_dim"]),
            "--detr_num_layers", str(p["layers"]), "--detr_num_heads", str(p["heads"]),
            "--detr_template_size", str(p["template"]),
        ]
    return common + ["--model_arch", "site_dia", "--dia_token_dim", str(p["token_dim"])]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--execute", action="store_true")
    ap.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--repo-root", default=str(Path(__file__).resolve().parents[1]))
    ap.add_argument("--output-root", default="outputs/detection_protocol")
    ap.add_argument("--data-root", default="")
    ap.add_argument("--site-label-dir", default="")
    ap.add_argument("--pretrained-ckpt", default="Pretrain_weight/best.pth")
    ap.add_argument("--deim-command", default=None)
    ap.add_argument("--python", default=sys.executable)
    args = ap.parse_args()
    repo = Path(args.repo_root).resolve()
    root = repo / args.output_root
    root.mkdir(parents=True, exist_ok=True)
    plan = full_detection_plan()
    plan_path = root / "detection_protocol_plan.jsonl"
    plan_path.write_text(
        "".join(json.dumps(c.to_dict(), sort_keys=True) + "\n" for c in plan), encoding="utf-8"
    )
    print(f"wrote {len(plan)}-run search plan to {plan_path} (216 DETR, 216 DEIM, 12 Site-AIT)")
    if not args.execute:
        print("dry run only: no training commands were launched")
        return
    if not args.data_root or not args.site_label_dir or not args.deim_command:
        ap.error("--execute requires --data-root, --site-label-dir, and --deim-command")

    winners = {}
    for candidate in plan:
        output = root / candidate.run_id
        output.mkdir(parents=True, exist_ok=True)
        result_path = output / "candidate_result.json"
        command = _command(candidate, output, args)
        if not result_path.exists() or not args.resume:
            env = dict(os.environ)
            env["IMG_DIR"] = args.data_root
            subprocess.run(command, cwd=repo, env=env, check=True)
        result = json.loads(result_path.read_text(encoding="utf-8"))
        score = float(result["val_ion_acc"])
        old = winners.get(candidate.family)
        if old is None or score > old["val_ion_acc"]:
            winners[candidate.family] = {**candidate.to_dict(), **result}
    (root / "search_winners.json").write_text(json.dumps(winners, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
