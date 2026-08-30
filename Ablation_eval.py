import argparse

import os

import time

from pathlib import Path


import cv2

import numpy as np

import torch


from nets.DWNetV2_unet import DWNetV2_unet

from Testset_eval import (

    DEFAULT_GT_ALL_BRIGHT_MASK,

    DEFAULT_MODEL_PATH,

    DEFAULT_TEST_ROOT,

    MASK_THRESHOLD,

    PRED_THRESHOLD,

    binary_average_precision,

    binary_auroc,

    calc_binary_metrics,

    collect_pairs,

    expected_calibration_error,

    extract_fixed_sites_from_gt,

    format_value,

    load_checkpoint_state,

    load_gray_image,

    make_state_labels_from_mask,

    make_table,

    preprocess_input,

    safe_nll,

)


DEFAULT_RESULT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs", "site_dia_ablation_result")


def ensure_dir(path):

    os.makedirs(path, exist_ok=True)


def instantiate_model(cfg, device):

    kwargs = cfg.get("model_kwargs", {})

    model = DWNetV2_unet(

        pre_trained=None,

        mode="eval",

        enable_site_dia=True,

        num_ions=300,

        dia_num_ion_attn_layers=kwargs.get("num_ion_attn_layers", 1),

        dia_use_global_cross_attn=kwargs.get("use_global_cross_attn", False),

        dia_disable_offset=kwargs.get("disable_offset", False),

        dia_disable_psf_mask=kwargs.get("disable_psf_mask", False),

        dia_disable_coord_head=kwargs.get("disable_coord_head", False),

    )

    state_dict = load_checkpoint_state(cfg["checkpoint"], device)

    state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}

    incompatible = model.load_state_dict(state_dict, strict=cfg["strict_load"])

    model.to(device)

    model.eval()

    return model, incompatible


def evaluate_variant(model, pairs, coords, coords_int, device, args):

    coords_t = torch.from_numpy(coords).float().unsqueeze(0).to(device)


    y_true_all = []

    y_prob_all = []

    y_pred_all = []

    uncertainty_all = []

    coord_l2_all = []

    count_abs_errors = []

    exact_count_matches = []

    bitstring_matches = []

    mask_dice = []

    mask_iou = []

    mask_f1 = []

    latency_ms = []

    true_counts = []


    with torch.no_grad():

        for idx, (img_path, mask_path) in enumerate(pairs, start=1):

            gray = load_gray_image(img_path)

            mask_img = load_gray_image(mask_path)

            gt_bin = (mask_img > MASK_THRESHOLD).astype(np.uint8)

            state_true = make_state_labels_from_mask(mask_img, coords_int)

            true_count = int(state_true.sum())

            true_counts.append(true_count)


            inp = preprocess_input(gray).to(device)

            if device.type == "cuda":

                torch.cuda.synchronize()

            t0 = time.perf_counter()

            out = model(inp, site_coords=coords_t)

            if device.type == "cuda":

                torch.cuda.synchronize()

            latency_ms.append((time.perf_counter() - t0) * 1000.0)


            state_prob = torch.sigmoid(out["bright_logit"]).squeeze(0).detach().cpu().numpy()

            state_pred = (state_prob >= args.state_threshold).astype(np.uint8)

            pred_count = int(state_pred.sum())

            count_abs_errors.append(abs(pred_count - true_count))

            exact_count_matches.append(int(pred_count == true_count))

            bitstring_matches.append(int(np.all(state_pred == state_true)))


            pred_coords = out["pred_coords"].squeeze(0).detach().cpu().numpy()

            coord_l2_all.append(np.sqrt(((pred_coords - coords) ** 2).sum(axis=1)))


            if "uncertainty" in out:

                uncertainty_all.append(out["uncertainty"].squeeze(0).detach().cpu().numpy())


            mask_prob = torch.sigmoid(out["mask_logits"]).squeeze().detach().cpu().numpy()

            pred_bin = (mask_prob >= args.mask_threshold).astype(np.uint8)

            mm = calc_binary_metrics(pred_bin, gt_bin)

            mask_dice.append(mm["dice"])

            mask_iou.append(mm["iou"])

            mask_f1.append(mm["f1"])


            y_true_all.append(state_true)

            y_prob_all.append(state_prob)

            y_pred_all.append(state_pred)


            if idx % args.log_every == 0 or idx == len(pairs):

                print(f"  [{idx}/{len(pairs)}] {img_path.name}")


    y_true = np.concatenate(y_true_all)

    y_prob = np.concatenate(y_prob_all)

    y_pred = np.concatenate(y_pred_all)

    state_metrics = calc_binary_metrics(y_pred, y_true)

    dark_metrics = calc_binary_metrics(1 - y_pred, 1 - y_true)

    coord_l2 = np.concatenate(coord_l2_all)

    latency_ms = np.asarray(latency_ms, dtype=np.float64)

    count_abs_errors = np.asarray(count_abs_errors, dtype=np.float64)

    uncertainty = np.concatenate(uncertainty_all) if uncertainty_all else np.asarray([], dtype=np.float64)


    return {

        "per_ion_acc": state_metrics["accuracy"],

        "bright_f1": state_metrics["f1"],

        "bright_precision": state_metrics["precision"],

        "bright_recall": state_metrics["recall"],

        "dark_recall": dark_metrics["recall"],

        "balanced_acc": 0.5 * (state_metrics["recall"] + state_metrics["specificity"]),

        "bitstring_exact": float(np.mean(bitstring_matches)),

        "count_mae": float(count_abs_errors.mean()),

        "exact_count_match": float(np.mean(exact_count_matches)),

        "state_count_fidelity": 1.0 - float(count_abs_errors.sum()) / max(1.0, float(np.sum(true_counts))),

        "mask_dice": float(np.mean(mask_dice)),

        "mask_iou": float(np.mean(mask_iou)),

        "mask_f1": float(np.mean(mask_f1)),

        "auroc": binary_auroc(y_true, y_prob),

        "auprc": binary_average_precision(y_true, y_prob),

        "brier": float(np.mean((y_prob - y_true) ** 2)),

        "nll": safe_nll(y_true, y_prob),

        "ece": expected_calibration_error(y_true, y_prob, n_bins=args.ece_bins),

        "coord_l2_mean": float(coord_l2.mean()),

        "coord_l2_p95": float(np.percentile(coord_l2, 95)),

        "latency_mean_ms": float(latency_ms.mean()),

        "latency_p95_ms": float(np.percentile(latency_ms, 95)),

        "uncertainty_mean": float(uncertainty.mean()) if uncertainty.size else float("nan"),

    }


def build_variant_configs(args):

    # Matches Table VI of the Site-AIT paper: one "full" row, an architectural

    # block (6 rows), and a PSF-reconstruction-consistency block (3 rows). The

    # three recon-block rows only differ from "full" in the training recipe

    # (loss configuration), not in model architecture, so they load with the

    # same model_kwargs as "full".

    all_rows = [

        {

            "key": "full",

            "name": "Site-AIT (full model)",

            "ckpt_attr": "full_ckpt",

            "model_kwargs": {"num_ion_attn_layers": args.full_ion_attn_layers},

            "description": "Retrained full Site-AIT model: physics pretraining + adaptive Top-16 tokenization + ion-token self-attention + PSF reconstruction consistency.",

        },

        {

            "key": "global_cross_attn",

            "name": "w/ global cross-attn",

            "ckpt_attr": "global_cross_attn_ckpt",

            "model_kwargs": {"use_global_cross_attn": True},

            "description": "Retrained with the adaptive Top-16 tokenizer replaced by a generic global cross-attention tokenizer.",

        },

        {

            "key": "no_self_attn",

            "name": "w/o ion self-attn",

            "ckpt_attr": "no_self_attn_ckpt",

            "model_kwargs": {"num_ion_attn_layers": 0},

            "description": "Retrained without ion-token self-attention (per-site classification, no cross-site context).",

        },

        {

            "key": "no_tok_offsets",

            "name": "w/o tokenization offsets",

            "ckpt_attr": "no_tok_offsets_ckpt",

            "model_kwargs": {"disable_offset": True},

            "description": "Retrained with the tokenization offset Delta^tok forced to 0 (nominal-center sampling only).",

        },

        {

            "key": "no_psf_mask",

            "name": "w/o learned PSF mask",

            "ckpt_attr": "no_psf_mask_ckpt",

            "model_kwargs": {"disable_psf_mask": True},

            "description": "Retrained with a uniform post-selection weight instead of the learnable PSF-inspired mask.",

        },

        {

            "key": "no_pretrain",

            "name": "w/o phys. pretraining",

            "ckpt_attr": "no_pretrain_ckpt",

            "model_kwargs": {"num_ion_attn_layers": args.full_ion_attn_layers},

            "description": "Retrained from scratch without physics-aware self-supervised initialization.",

        },

        {

            "key": "no_coord_constraint",

            "name": "w/o learn. coord. constraint",

            "ckpt_attr": "no_coord_constraint_ckpt",

            "model_kwargs": {"disable_coord_head": True},

            "description": "Retrained with the coordinate head removed; predicted coordinates are always the nominal lattice.",

        },

        {

            "key": "no_recon_loss",

            "name": "Full - L_recon, no recon. loss",

            "ckpt_attr": "no_recon_loss_ckpt",

            "model_kwargs": {"num_ion_attn_layers": args.full_ion_attn_layers},

            "description": "Retrained with the PSF reconstruction consistency loss disabled entirely.",

        },

        {

            "key": "recon_sigma_frozen",

            "name": "Full + L_recon, sigma_psf frozen",

            "ckpt_attr": "recon_sigma_frozen_ckpt",

            "model_kwargs": {"num_ion_attn_layers": args.full_ion_attn_layers},

            "description": "Retrained with the PSF reconstruction consistency loss active but sigma_psf not optimized.",

        },

        {

            "key": "recon_no_warmup",

            "name": "Full + L_recon, no warm-up",

            "ckpt_attr": "recon_no_warmup_ckpt",

            "model_kwargs": {"num_ion_attn_layers": args.full_ion_attn_layers},

            "description": "Retrained with lambda_recon at its maximum from epoch 0 and no state-head gradient detachment.",

        },

    ]


    selected_keys = {k.strip() for k in args.only.split(",") if k.strip()} or None

    if selected_keys:

        unknown = selected_keys - {row["key"] for row in all_rows}

        if unknown:

            raise ValueError(f"Unknown variant(s) in --only: {sorted(unknown)}")

        all_rows = [row for row in all_rows if row["key"] in selected_keys]


    missing = [f"--{row['ckpt_attr']}" for row in all_rows if not getattr(args, row["ckpt_attr"])]

    if missing:

        raise ValueError(

            "Formal retrained ablation requires separate checkpoints for every selected variant. "

            f"Missing arguments: {', '.join(missing)}.\n"

            "Train each variant first with train_retrained_ablation.py, or narrow the set with --only, "

            "or pass the correct checkpoint paths. Do not reuse the full checkpoint for ablation rows."

        )


    rows = []

    for row in all_rows:

        checkpoint = getattr(args, row["ckpt_attr"])

        rows.append(

            {

                "name": row["name"],

                "checkpoint": checkpoint,

                "strict_load": True,

                "model_kwargs": row["model_kwargs"],

                "description": row["description"],

            }

        )


    not_found = [(row["name"], row["checkpoint"]) for row in rows if not os.path.exists(row["checkpoint"])]

    if not_found:

        details = "\n".join(f"  - {name}: {path}" for name, path in not_found)

        raise FileNotFoundError(

            "Some retrained ablation checkpoints do not exist:\n"

            f"{details}\n"

            "Run train_retrained_ablation.py first or pass the correct checkpoint paths."

        )

    return rows


def render_ablation_table(results):

    columns = [

        ("Variant", "name"),

        ("Per-ion Acc.", "per_ion_acc"),

        ("Bright F1", "bright_f1"),

        ("Dark Recall", "dark_recall"),

        ("Bitstring", "bitstring_exact"),

        ("Count MAE", "count_mae"),

        ("Mask Dice", "mask_dice"),

        ("ECE", "ece"),

        ("Coord L2", "coord_l2_mean"),

        ("Latency ms", "latency_mean_ms"),

    ]

    header = [c[0] for c in columns]

    table_rows = []

    for r in results:

        table_rows.append([r["name"]] + [format_value(r.get(k)) for _, k in columns[1:]])

    widths = [len(h) for h in header]

    for row in table_rows:

        for i, cell in enumerate(row):

            widths[i] = max(widths[i], len(str(cell)))

    sep = "+" + "+".join("-" * (w + 2) for w in widths) + "+"

    lines = [sep]

    lines.append("| " + " | ".join(header[i].ljust(widths[i]) for i in range(len(header))) + " |")

    lines.append(sep)

    for row in table_rows:

        lines.append("| " + " | ".join(str(row[i]).ljust(widths[i]) for i in range(len(header))) + " |")

    lines.append(sep)

    return "\n".join(lines)


def main():

    args = parse_args()

    ensure_dir(args.result_dir)

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))

    pairs = collect_pairs(args.test_root)

    if args.max_samples > 0:

        pairs = pairs[: args.max_samples]

    coords, coords_int, _, _ = extract_fixed_sites_from_gt(args.gt_all_bright_mask)


    variants = build_variant_configs(args)

    results = []

    for cfg in variants:

        if not os.path.exists(cfg["checkpoint"]):

            print(f"Skipping {cfg['name']}: checkpoint not found: {cfg['checkpoint']}")

            continue

        print("=" * 80)

        print(f"Evaluating {cfg['name']}")

        print(f"Checkpoint: {cfg['checkpoint']}")

        model, incompatible = instantiate_model(cfg, device)

        metrics = evaluate_variant(model, pairs, coords, coords_int, device, args)

        metrics.update(

            {

                "name": cfg["name"],

                "checkpoint": cfg["checkpoint"],

                "description": cfg["description"],

                "missing_keys": len(getattr(incompatible, "missing_keys", [])),

                "unexpected_keys": len(getattr(incompatible, "unexpected_keys", [])),

            }

        )

        results.append(metrics)

        del model

        if torch.cuda.is_available():

            torch.cuda.empty_cache()


    if not results:

        raise RuntimeError("No ablation variant was evaluated. Check checkpoint paths.")


    table = render_ablation_table(results)

    detail_rows = []

    for r in results:

        detail_rows.extend(

            [

                (f"{r['name']} checkpoint", r["checkpoint"], r["description"]),

                (f"{r['name']} AUROC", r["auroc"], "Threshold-free bright/dark ranking metric."),

                (f"{r['name']} AUPRC", r["auprc"], "Average precision for bright-state prediction."),

                (f"{r['name']} NLL", r["nll"], "Negative log-likelihood; lower is better."),

                (f"{r['name']} Brier", r["brier"], "Probability mean-squared error; lower is better."),

                (f"{r['name']} Count fidelity", r["state_count_fidelity"], "1 - total absolute count error / total true bright count."),

                (f"{r['name']} load missing/unexpected", f"{r['missing_keys']}/{r['unexpected_keys']}", "Nonzero values mean partial-load diagnostic ablation rather than a separately trained checkpoint."),

            ]

        )


    detail_table = make_table(detail_rows)

    report = f"""
Site-DIA ablation study
=======================
Test root        : {args.test_root}
GT site mask     : {args.gt_all_bright_mask}
Evaluated samples: {len(pairs)}
Output file      : {Path(args.result_dir) / args.report_name}

Main ablation table
-------------------
{table}

Detailed metrics and notes
--------------------------
{detail_table}

Explanation
-----------
This ablation matches Table VI of the Site-AIT paper. Architectural block:
1. w/ global cross-attn: replaces adaptive Top-16 tokenization with generic global cross-attention.
2. w/o ion self-attn: removes cross-site joint inference (per-site classification only).
3. w/o tokenization offsets: forces the frame-specific recentering offset Delta^tok to 0.
4. w/o learned PSF mask: replaces the learnable post-selection PSF weight with a uniform one.
5. w/o phys. pretraining: trains from scratch, without physics-aware self-supervised initialization.
6. w/o learn. coord. constraint: removes the coordinate refinement head entirely.
PSF reconstruction consistency block:
7. Full - L_recon: disables the reconstruction consistency loss entirely.
8. sigma_psf frozen: keeps L_recon active but does not optimize the PSF width.
9. no warm-up: uses the maximum L_recon weight from epoch 0, without the warm-up schedule or state-head gradient detachment.

This table is a formal retrained ablation: each row is loaded from its own separately trained checkpoint. No row reuses the full checkpoint with modules switched off only at inference time.
"""

    out_path = Path(args.result_dir) / args.report_name

    with open(out_path, "w", encoding="utf-8") as f:

        f.write(report)

    print("\nAblation finished.")

    print(f"Report: {out_path}")

    print(table)


def parse_args():

    parser = argparse.ArgumentParser(description="Ablation evaluation for Site-DIA on intersection_test_data.")

    parser.add_argument("--model_path", type=str, default=DEFAULT_MODEL_PATH, help="Deprecated; use --full_ckpt for formal retrained ablation.")

    parser.add_argument("--full_ckpt", type=str, default="", help="Checkpoint for 'Site-AIT (full model)'.")

    parser.add_argument("--global_cross_attn_ckpt", type=str, default="", help="Checkpoint for the 'w/ global cross-attn' row.")

    parser.add_argument("--no_self_attn_ckpt", type=str, default="", help="Checkpoint for the 'w/o ion self-attn' row.")

    parser.add_argument("--no_tok_offsets_ckpt", type=str, default="", help="Checkpoint for the 'w/o tokenization offsets' row.")

    parser.add_argument("--no_psf_mask_ckpt", type=str, default="", help="Checkpoint for the 'w/o learned PSF mask' row.")

    parser.add_argument("--no_pretrain_ckpt", type=str, default="", help="Checkpoint for the 'w/o phys. pretraining' row.")

    parser.add_argument("--no_coord_constraint_ckpt", type=str, default="", help="Checkpoint for the 'w/o learn. coord. constraint' row.")

    parser.add_argument("--no_recon_loss_ckpt", type=str, default="", help="Checkpoint for the 'Full - L_recon, no recon. loss' row.")

    parser.add_argument("--recon_sigma_frozen_ckpt", type=str, default="", help="Checkpoint for the 'Full + L_recon, sigma_psf frozen' row.")

    parser.add_argument("--recon_no_warmup_ckpt", type=str, default="", help="Checkpoint for the 'Full + L_recon, no warm-up' row.")

    parser.add_argument(

        "--only",

        type=str,

        default="",

        help=(

            "Comma-separated subset of variant keys to evaluate (default: all 10): "

            "full,global_cross_attn,no_self_attn,no_tok_offsets,no_psf_mask,no_pretrain,"

            "no_coord_constraint,no_recon_loss,recon_sigma_frozen,recon_no_warmup"

        ),

    )

    parser.add_argument("--test_root", type=str, default=DEFAULT_TEST_ROOT)

    parser.add_argument("--gt_all_bright_mask", type=str, default=DEFAULT_GT_ALL_BRIGHT_MASK)

    parser.add_argument("--result_dir", type=str, default=DEFAULT_RESULT_DIR)

    parser.add_argument("--device", type=str, default="", help="cuda, cpu, or empty for auto")

    parser.add_argument("--report_name", type=str, default="ablation.txt")

    parser.add_argument("--mask_threshold", type=float, default=PRED_THRESHOLD)

    parser.add_argument("--state_threshold", type=float, default=PRED_THRESHOLD)

    parser.add_argument("--ece_bins", type=int, default=15)

    parser.add_argument("--full_ion_attn_layers", type=int, default=1)

    parser.add_argument("--strict_load", action="store_true", help="Strictly load full checkpoints. Default is non-strict for convenience.")

    parser.add_argument("--strict_load_if_ckpt_provided", action="store_true", help="Use strict loading when a separate variant checkpoint is provided.")

    parser.add_argument("--max_samples", type=int, default=0, help="Debug option: evaluate only first N samples if >0.")

    parser.add_argument("--log_every", type=int, default=200)

    return parser.parse_args()


if __name__ == "__main__":

    main()

