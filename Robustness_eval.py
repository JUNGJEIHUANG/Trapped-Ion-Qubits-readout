"""Physical-robustness evaluation under the controlled perturbation operators

P1-P6 (Site-AIT paper Sec. V.C, App. I). Applies each operator at each of its

paper-specified levels to the test set, evaluates a trained checkpoint (zero-

shot, no retraining or fine-tuning, matching the paper's protocol), and writes

a long-format CSV (perturbation, level, metric, mean, std) with mean/std taken

over the requested seeds -- the exact schema plot_table3_robustness.py expects.

"""

import argparse

import os

from pathlib import Path


import numpy as np

import pandas as pd

import torch


from perturbations import (

    PERTURBATION_LEVELS,

    p1_coordinate_drift,

    p2_psf_broadening,

    p3_snr_reduction,

    p4_hot_pixel_injection,

    p5_single_site_rearrangement,

    p6_anisotropic_psf_elongation,

)

from Testset_eval import (

    DEFAULT_GT_ALL_BRIGHT_MASK,

    DEFAULT_MODEL_PATH,

    DEFAULT_TEST_ROOT,

    calc_binary_metrics,

    collect_pairs,

    extract_fixed_sites_from_gt,

    load_gray_image,

    load_model,

    make_state_labels_from_mask,

)


DEFAULT_RESULT_DIR = os.path.join(

    os.path.dirname(os.path.abspath(__file__)), "outputs", "robustness_eval"

)


ALL_PERTURBATIONS = ("P1", "P2", "P3", "P4", "P5", "P6")


def compute_loc_error(pred_coords, ref_coords):

    """Site-AIT paper App. 'Metric definitions': the dispersion of the

    coordinate-refinement residual around its own mean, pooled over all valid

    sites -- a precision (reproducibility), not accuracy, measure.

    """

    delta = pred_coords - ref_coords

    delta_bar = delta.mean(axis=0)

    centered = delta - delta_bar

    return float(np.sqrt((centered**2).sum(axis=1).mean()))


def apply_perturbation(name, level, image_f, coords, rng):

    """Returns (image_for_model, coords_for_model). `coords` is never mutated

    in place; P5 also perturbs the image but leaves the nominal lattice fed to

    the model unchanged, per the paper."""

    if name == "P1":

        return image_f, p1_coordinate_drift(coords, level, rng=rng)

    if name == "P2":

        return p2_psf_broadening(image_f, level), coords

    if name == "P3":

        return p3_snr_reduction(image_f, level, rng=rng), coords

    if name == "P4":

        return p4_hot_pixel_injection(image_f, level, saturation_value=1.0, rng=rng), coords

    if name == "P5":

        valid = np.ones(coords.shape[0], dtype=np.float32)

        perturbed_image, _ = p5_single_site_rearrangement(image_f, coords, valid, level, rng=rng)

        return perturbed_image, coords

    if name == "P6":

        return p6_anisotropic_psf_elongation(image_f, level, sigma_x=1.0), coords

    raise ValueError(f"Unknown perturbation: {name}")


def evaluate_one_pass(model, pairs, coords, coords_int, device, perturbation, level, rng, state_threshold, max_samples):

    y_true_all = []

    y_prob_all = []

    pred_coords_all = []


    sample_pairs = pairs[:max_samples] if max_samples > 0 else pairs


    with torch.no_grad():

        for img_path, mask_path in sample_pairs:

            gray = load_gray_image(img_path)

            mask_img = load_gray_image(mask_path)

            state_true = make_state_labels_from_mask(mask_img, coords_int)


            image_f = gray.astype(np.float32) / 255.0

            image_f, use_coords = apply_perturbation(perturbation, level, image_f, coords, rng)


            inp = torch.from_numpy(np.ascontiguousarray(image_f)).float().unsqueeze(0).unsqueeze(0).to(device)

            coords_t = torch.from_numpy(use_coords).float().unsqueeze(0).to(device)


            out = model(inp, site_coords=coords_t)

            state_prob = torch.sigmoid(out["bright_logit"]).squeeze(0).detach().cpu().numpy()

            pred_coords = out["pred_coords"].squeeze(0).detach().cpu().numpy()


            y_true_all.append(state_true)

            y_prob_all.append(state_prob)

            pred_coords_all.append(pred_coords)


    y_true = np.concatenate(y_true_all)

    y_prob = np.concatenate(y_prob_all)

    y_pred = (y_prob >= state_threshold).astype(np.uint8)


    m = calc_binary_metrics(y_pred, y_true)

    balanced_acc = 0.5 * (m["recall"] + m["specificity"])


    # Reference is always the true (unperturbed) nominal lattice, since loc.
    # error measures the reproducibility of the refined coordinate, not the
    # deliberately drifted input fed to the model under P1.
    pred_coords_pooled = np.concatenate(pred_coords_all, axis=0)

    ref_coords_pooled = np.concatenate([coords for _ in pred_coords_all], axis=0)

    loc_error = compute_loc_error(pred_coords_pooled, ref_coords_pooled)


    return {

        "accuracy": m["accuracy"],

        "balanced_acc": balanced_acc,

        "precision": m["precision"],

        "recall": m["recall"],

        "f1": m["f1"],

        "loc_error": loc_error,

    }


def run(args):

    os.makedirs(args.result_dir, exist_ok=True)

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))


    pairs = collect_pairs(args.test_root)

    coords, coords_int, _, _ = extract_fixed_sites_from_gt(args.gt_all_bright_mask)


    model, incompatible = load_model(

        args.model_path,

        device,

        model_arch=args.model_arch,

        allow_partial_load=args.allow_partial_load,

        num_ion_attn_layers=args.num_ion_attn_layers,

    )

    print(f"Loaded {args.model_arch} from {args.model_path} "

          f"(missing={len(incompatible.missing_keys)}, unexpected={len(incompatible.unexpected_keys)})")


    perturbations = [p.strip().upper() for p in args.perturbations.split(",") if p.strip()]

    unknown = [p for p in perturbations if p not in ALL_PERTURBATIONS]

    if unknown:

        raise ValueError(f"Unknown perturbation(s): {unknown}. Valid: {ALL_PERTURBATIONS}")


    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]


    rows = []

    for perturbation in perturbations:

        levels = PERTURBATION_LEVELS[perturbation]

        for level in levels:

            per_seed = {"accuracy": [], "balanced_acc": [], "precision": [], "recall": [], "f1": [], "loc_error": []}

            for seed in seeds:

                rng = np.random.default_rng(

                    [seed, hash(perturbation) & 0xFFFF, int(round(level * 1000))]

                )

                metrics = evaluate_one_pass(

                    model, pairs, coords, coords_int, device, perturbation, level, rng,

                    args.state_threshold, args.max_samples,

                )

                for k, v in metrics.items():

                    per_seed[k].append(v)

                print(f"  {perturbation} level={level} seed={seed}: "

                      f"acc={metrics['accuracy']:.4f} f1={metrics['f1']:.4f} loc_err={metrics['loc_error']:.4f}")


            for metric, values in per_seed.items():

                values = np.asarray(values, dtype=np.float64)

                rows.append(

                    {

                        "perturbation": perturbation,

                        "level": level,

                        "metric": metric,

                        "mean": float(values.mean()),

                        "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,

                    }

                )


    df = pd.DataFrame(rows)

    out_path = Path(args.result_dir) / args.out_csv

    df.to_csv(out_path, index=False)

    print(f"\nWrote robustness results: {out_path}")

    return df


def parse_args():

    parser = argparse.ArgumentParser(

        description="Evaluate a Site-AIT-family checkpoint under P1-P6 controlled perturbations."

    )

    parser.add_argument("--model_path", type=str, default=DEFAULT_MODEL_PATH)

    parser.add_argument("--model_arch", type=str, default="site_dia", choices=["site_dia", "detr_query", "set_transformer"])

    parser.add_argument("--test_root", type=str, default=DEFAULT_TEST_ROOT)

    parser.add_argument("--gt_all_bright_mask", type=str, default=DEFAULT_GT_ALL_BRIGHT_MASK)

    parser.add_argument("--result_dir", type=str, default=DEFAULT_RESULT_DIR)

    parser.add_argument("--out_csv", type=str, default="table3_robustness_measured.csv")

    parser.add_argument("--device", type=str, default="", help="cuda, cpu, or empty for auto")

    parser.add_argument("--state_threshold", type=float, default=0.5)

    parser.add_argument("--num_ion_attn_layers", type=int, default=1)

    parser.add_argument("--allow_partial_load", action="store_true")

    parser.add_argument("--perturbations", type=str, default=",".join(ALL_PERTURBATIONS))

    parser.add_argument("--seeds", type=str, default="1,2,3")

    parser.add_argument("--max_samples", type=int, default=0, help="Debug option: evaluate only first N samples if >0.")

    return parser.parse_args()


if __name__ == "__main__":

    run(parse_args())

