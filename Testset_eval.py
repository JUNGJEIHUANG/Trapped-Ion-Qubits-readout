import argparse

import math

import os

import time

from glob import glob

from pathlib import Path


import cv2

import numpy as np

import torch

import torch.nn.functional as F


from nets.DWNetV2_unet import DWNetV2_unet

from nets.DETRQuery import DETRStyleQuery

from nets.SetTransformer import SetTransformerReadout

from nets.StandardUNet import StandardUNet

from nets.ViTUNet import ViTUNet

from nets.SETR import SETR

from nets.SegFormer import SegFormer

from nets.Segmenter import Segmenter


_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

_DATA_ROOT = os.environ.get("DATA_ROOT", os.path.join(_PROJECT_ROOT, "data"))


DEFAULT_MODEL_PATH = os.environ.get(

    "MODEL_PATH",

    os.path.join(_PROJECT_ROOT, "outputs", "site_dia_stage3", "sample_50000", "best.pth"),

)

DEFAULT_TEST_ROOT = os.path.join(_DATA_ROOT, "intersection_test_data")

DEFAULT_GT_ALL_BRIGHT_MASK = os.path.join(_DATA_ROOT, "Ground Truth", "Ground Truth.png")

DEFAULT_RESULT_DIR = os.path.join(_PROJECT_ROOT, "This_Work", "SiteDIA_Test_Result")


MASK_THRESHOLD = 128

PRED_THRESHOLD = 0.5

MIN_COMPONENT_AREA = 1

VALID_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def ensure_dir(path):

    os.makedirs(path, exist_ok=True)


def load_gray_image(path):

    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)

    if img is None:

        raise ValueError(f"Failed to read image: {path}")

    return img


def preprocess_input(gray_img):

    tensor = torch.from_numpy(np.ascontiguousarray(gray_img)).float().div(255.0)

    return tensor.unsqueeze(0).unsqueeze(0)


def collect_pairs(test_root):

    test_root = Path(test_root)

    image_dir = test_root / "images"

    mask_dir = test_root / "masks"

    if not image_dir.is_dir():

        raise FileNotFoundError(f"Image directory not found: {image_dir}")

    if not mask_dir.is_dir():

        raise FileNotFoundError(f"Mask directory not found: {mask_dir}")


    mask_files = sorted(mask_dir.glob("*.png"))

    if not mask_files:

        raise FileNotFoundError(f"No PNG masks found in: {mask_dir}")


    pairs = []

    missing = []

    for mask_path in mask_files:

        stem = mask_path.stem

        image_path = None

        for ext in VALID_EXTS:

            candidate = image_dir / f"{stem}{ext}"

            if candidate.exists():

                image_path = candidate

                break

        if image_path is None:

            missing.append(mask_path.name)

        else:

            pairs.append((image_path, mask_path))


    if missing:

        print(f"Warning: {len(missing)} masks have no matching image. First 10: {missing[:10]}")

    if not pairs:

        raise RuntimeError(f"No valid image-mask pairs under {test_root}")

    return pairs


def extract_fixed_sites_from_gt(gt_mask_path, threshold=MASK_THRESHOLD, min_area=MIN_COMPONENT_AREA):

    gt = load_gray_image(gt_mask_path)

    binary = (gt > threshold).astype(np.uint8)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)


    sites = []

    for label_id in range(1, num_labels):

        area = int(stats[label_id, cv2.CC_STAT_AREA])

        if area < min_area:

            continue

        cx, cy = centroids[label_id]


        x_int = int(np.floor(float(cx) + 0.5))

        y_int = int(np.floor(float(cy) + 0.5))

        x_int = min(max(x_int, 0), gt.shape[1] - 1)

        y_int = min(max(y_int, 0), gt.shape[0] - 1)

        sites.append((float(cx), float(cy), x_int, y_int, area))


    sites = sorted(sites, key=lambda s: (s[1], s[0]))

    coords = np.asarray([[s[0], s[1]] for s in sites], dtype=np.float32)

    coords_int = np.asarray([[s[2], s[3]] for s in sites], dtype=np.int64)

    areas = np.asarray([s[4] for s in sites], dtype=np.int64)

    return coords, coords_int, areas, gt.shape


def make_state_labels_from_mask(mask_img, coords_int, threshold=MASK_THRESHOLD):

    ys = coords_int[:, 1]

    xs = coords_int[:, 0]

    return (mask_img[ys, xs] > threshold).astype(np.uint8)


def calc_binary_metrics(pred, target, eps=1e-9):

    pred = pred.astype(bool)

    target = target.astype(bool)

    tp = np.logical_and(pred, target).sum(dtype=np.float64)

    tn = np.logical_and(~pred, ~target).sum(dtype=np.float64)

    fp = np.logical_and(pred, ~target).sum(dtype=np.float64)

    fn = np.logical_and(~pred, target).sum(dtype=np.float64)

    precision = tp / (tp + fp + eps)

    recall = tp / (tp + fn + eps)

    specificity = tn / (tn + fp + eps)

    f1 = 2.0 * precision * recall / (precision + recall + eps)

    acc = (tp + tn) / (tp + tn + fp + fn + eps)

    iou = tp / (tp + fp + fn + eps)

    dice = 2.0 * tp / (2.0 * tp + fp + fn + eps)

    return {

        "tp": tp,

        "tn": tn,

        "fp": fp,

        "fn": fn,

        "precision": precision,

        "recall": recall,

        "specificity": specificity,

        "f1": f1,

        "accuracy": acc,

        "iou": iou,

        "dice": dice,

    }


def binary_auroc(y_true, y_score):

    y_true = np.asarray(y_true).astype(np.int64)

    y_score = np.asarray(y_score).astype(np.float64)

    pos = y_true == 1

    neg = y_true == 0

    n_pos = int(pos.sum())

    n_neg = int(neg.sum())

    if n_pos == 0 or n_neg == 0:

        return float("nan")

    order = np.argsort(y_score)

    ranks = np.empty_like(order, dtype=np.float64)

    ranks[order] = np.arange(1, len(y_score) + 1)


    sorted_scores = y_score[order]

    start = 0

    while start < len(sorted_scores):

        end = start + 1

        while end < len(sorted_scores) and sorted_scores[end] == sorted_scores[start]:

            end += 1

        if end - start > 1:

            avg_rank = (start + 1 + end) / 2.0

            ranks[order[start:end]] = avg_rank

        start = end

    sum_pos_ranks = ranks[pos].sum()

    return float((sum_pos_ranks - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def binary_average_precision(y_true, y_score):

    y_true = np.asarray(y_true).astype(np.int64)

    y_score = np.asarray(y_score).astype(np.float64)

    n_pos = int((y_true == 1).sum())

    if n_pos == 0:

        return float("nan")

    order = np.argsort(-y_score)

    y_sorted = y_true[order]

    tp = np.cumsum(y_sorted == 1)

    fp = np.cumsum(y_sorted == 0)

    precision = tp / np.maximum(tp + fp, 1)

    return float((precision * (y_sorted == 1)).sum() / n_pos)


def expected_calibration_error(y_true, y_prob, n_bins=15):

    y_true = np.asarray(y_true).astype(np.int64)

    y_prob = np.asarray(y_prob).astype(np.float64)

    pred = (y_prob >= 0.5).astype(np.int64)

    conf = np.maximum(y_prob, 1.0 - y_prob)

    correct = (pred == y_true).astype(np.float64)

    ece = 0.0

    for lo in np.linspace(0.0, 1.0, n_bins, endpoint=False):

        hi = lo + 1.0 / n_bins

        if hi >= 1.0:

            in_bin = (conf >= lo) & (conf <= hi)

        else:

            in_bin = (conf >= lo) & (conf < hi)

        if not np.any(in_bin):

            continue

        ece += in_bin.mean() * abs(correct[in_bin].mean() - conf[in_bin].mean())

    return float(ece)


def safe_nll(y_true, y_prob, eps=1e-7):

    y_true = np.asarray(y_true).astype(np.float64)

    y_prob = np.asarray(y_prob).astype(np.float64)

    y_prob = np.clip(y_prob, eps, 1.0 - eps)

    return float(-(y_true * np.log(y_prob) + (1 - y_true) * np.log(1 - y_prob)).mean())


def count_components(binary_mask, min_area=MIN_COMPONENT_AREA):

    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(binary_mask.astype(np.uint8), connectivity=8)

    count = 0

    for label_id in range(1, num_labels):

        if stats[label_id, cv2.CC_STAT_AREA] >= min_area:

            count += 1

    return count


def load_checkpoint_state(model_path, device):

    ckpt = torch.load(model_path, map_location=device)

    if isinstance(ckpt, dict):

        for key in ("model_state_dict", "state_dict", "model"):

            if key in ckpt and isinstance(ckpt[key], dict):

                return ckpt[key]

    return ckpt


def load_model(model_path, device, model_arch="site_dia", allow_partial_load=False, num_ion_attn_layers=1):

    if model_arch == "site_dia":

        model = DWNetV2_unet(

            pre_trained=None,

            mode="eval",

            enable_site_dia=True,

            num_ions=300,

            dia_num_ion_attn_layers=num_ion_attn_layers,

        )

    elif model_arch == "dwunet":

        model = DWNetV2_unet(pre_trained=None, mode="eval")

    elif model_arch == "detr_query":

        model = DETRStyleQuery(num_ions=300)

    elif model_arch == "set_transformer":

        model = SetTransformerReadout(num_ions=300)

    elif model_arch == "standard_unet":

        model = StandardUNet()

    elif model_arch == "vit_unet":

        model = ViTUNet()

    elif model_arch == "setr":

        model = SETR()

    elif model_arch == "segformer":

        model = SegFormer()

    elif model_arch == "segmenter":

        model = Segmenter()

    else:

        raise ValueError(f"Unsupported model_arch: {model_arch}")


    state_dict = load_checkpoint_state(model_path, device)

    state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}

    incompatible = model.load_state_dict(state_dict, strict=not allow_partial_load)

    model.to(device)

    model.eval()

    return model, incompatible


def format_value(v):

    if v is None:

        return "-"

    if isinstance(v, str):

        return v

    if isinstance(v, (int, np.integer)):

        return str(int(v))

    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):

        return str(v)

    return f"{float(v):.6f}"


def make_table(rows):

    headers = ["Metric", "Value", "Explanation"]

    str_rows = [[str(r[0]), format_value(r[1]), str(r[2])] for r in rows]

    widths = [len(h) for h in headers]

    for row in str_rows:

        for i, cell in enumerate(row):

            widths[i] = max(widths[i], len(cell))

    sep = "+" + "+".join("-" * (w + 2) for w in widths) + "+"

    lines = [sep]

    lines.append("| " + " | ".join(headers[i].ljust(widths[i]) for i in range(3)) + " |")

    lines.append(sep)

    for row in str_rows:

        lines.append("| " + " | ".join(row[i].ljust(widths[i]) for i in range(3)) + " |")

    lines.append(sep)

    return "\n".join(lines)


def evaluate(args):

    ensure_dir(args.result_dir)

    result_dir = Path(args.result_dir)

    pairs = collect_pairs(args.test_root)

    coords, coords_int, site_areas, image_shape = extract_fixed_sites_from_gt(args.gt_all_bright_mask)

    num_ions = coords.shape[0]

    if num_ions != 300:

        print(f"Warning: extracted {num_ions} ion sites from GT mask, expected 300.")


    device = torch.device("cuda")

    if not os.path.exists(args.model_path):

        raise FileNotFoundError(

            f"Model checkpoint not found: {args.model_path}\n"

            "Pass --model_path to a trained Site-DIA checkpoint."

        )

    model, incompatible = load_model(

        args.model_path,

        device,

        model_arch=args.model_arch,

        allow_partial_load=args.allow_partial_load,

        num_ion_attn_layers=args.num_ion_attn_layers,

    )

    total_params = sum(p.numel() for p in model.parameters())

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)


    coords_t = torch.from_numpy(coords).float().unsqueeze(0).to(device)


    all_state_true = []

    all_state_prob = []

    all_state_pred = []

    all_uncertainty = []

    all_exist_prob = []

    all_coord_l2 = []

    all_coord_l1 = []

    mask_metric_rows = []

    count_abs_errors = []

    count_signed_errors = []

    true_counts = []

    exact_count_matches = []

    bitstring_matches = []

    inference_times_ms = []

    per_sample_lines = []


    with torch.no_grad():

        for idx, (img_path, mask_path) in enumerate(pairs, start=1):

            gray = load_gray_image(img_path)

            mask_img = load_gray_image(mask_path)

            gt_bin = (mask_img > MASK_THRESHOLD).astype(np.uint8)

            state_true = make_state_labels_from_mask(mask_img, coords_int)


            inp = preprocess_input(gray).to(device)

            if device.type == "cuda":

                torch.cuda.synchronize()

            t0 = time.perf_counter()

            if args.model_arch in ("site_dia", "detr_query", "set_transformer"):

                out = model(inp, site_coords=coords_t)

                mask_logits = out["mask_logits"]

                state_prob = torch.sigmoid(out["bright_logit"]).squeeze(0).detach().cpu().numpy()

                state_pred = (state_prob >= args.state_threshold).astype(np.uint8)

                pred_coords = out["pred_coords"].squeeze(0).detach().cpu().numpy()

                coord_diff = pred_coords - coords

                coord_l2 = np.sqrt((coord_diff ** 2).sum(axis=1))

                coord_l1 = np.abs(coord_diff).sum(axis=1)

                uncertainty = out.get("uncertainty", None)

                if uncertainty is not None:

                    uncertainty = uncertainty.squeeze(0).detach().cpu().numpy()

                exist_logit = out.get("exist_logit", None)

                if exist_logit is not None:

                    exist_prob = torch.sigmoid(exist_logit).squeeze(0).detach().cpu().numpy()

                else:

                    exist_prob = np.ones_like(state_prob)

            else:

                mask_logits = model(inp)

                state_prob = None

                state_pred = None

                coord_l2 = None

                coord_l1 = None

                uncertainty = None

                exist_prob = None


            if device.type == "cuda":

                torch.cuda.synchronize()

            infer_ms = (time.perf_counter() - t0) * 1000.0

            inference_times_ms.append(infer_ms)


            mask_prob = torch.sigmoid(mask_logits).squeeze().detach().cpu().numpy()

            pred_bin = (mask_prob >= args.mask_threshold).astype(np.uint8)

            m = calc_binary_metrics(pred_bin, gt_bin)

            mask_metric_rows.append(m)


            if state_prob is None:

                _ys = coords_int[:, 1].clip(0, mask_prob.shape[0] - 1)

                _xs = coords_int[:, 0].clip(0, mask_prob.shape[1] - 1)

                state_prob = mask_prob[_ys, _xs].astype(np.float32)

                state_pred = (state_prob >= args.state_threshold).astype(np.uint8)

                exist_prob = np.ones_like(state_prob)


                coord_l2 = np.zeros(num_ions, dtype=np.float32)

                coord_l1 = np.zeros(num_ions, dtype=np.float32)


            if args.save_pred_masks:

                out_mask_path = result_dir / "pred_masks" / f"{img_path.stem}_pred.png"

                ensure_dir(out_mask_path.parent)

                cv2.imwrite(str(out_mask_path), pred_bin * 255)


            gt_count = int(state_true.sum())

            true_counts.append(gt_count)

            if state_prob is not None:

                pred_count = int(state_pred.sum())

                count_err = pred_count - gt_count

                count_signed_errors.append(count_err)

                count_abs_errors.append(abs(count_err))

                exact_count_matches.append(int(pred_count == gt_count))

                bitstring_matches.append(int(np.all(state_pred == state_true)))

                all_state_true.append(state_true)

                all_state_prob.append(state_prob)

                all_state_pred.append(state_pred)

                all_exist_prob.append(exist_prob)

                if uncertainty is not None:

                    all_uncertainty.append(uncertainty)

                all_coord_l2.append(coord_l2)

                all_coord_l1.append(coord_l1)

                per_sample_lines.append(

                    f"{idx},{img_path.name},{gt_count},{pred_count},{count_err},{m['dice']:.6f},{m['iou']:.6f},{infer_ms:.4f}"

                )

            else:

                pred_count = count_components(pred_bin)

                gt_components = count_components(gt_bin)

                count_err = pred_count - gt_components

                count_signed_errors.append(count_err)

                count_abs_errors.append(abs(count_err))

                exact_count_matches.append(int(pred_count == gt_components))

                per_sample_lines.append(

                    f"{idx},{img_path.name},{gt_components},{pred_count},{count_err},{m['dice']:.6f},{m['iou']:.6f},{infer_ms:.4f}"

                )


            if idx % args.log_every == 0 or idx == len(pairs):

                print(f"[{idx}/{len(pairs)}] {img_path.name} Dice={m['dice']:.4f} IoU={m['iou']:.4f} CountErr={count_err} Time={infer_ms:.2f} ms")


    mask_mean = {k: float(np.mean([m[k] for m in mask_metric_rows])) for k in mask_metric_rows[0] if k not in ("tp", "tn", "fp", "fn")}

    mask_sum = {k: float(np.sum([m[k] for m in mask_metric_rows])) for k in ("tp", "tn", "fp", "fn")}


    count_abs_errors = np.asarray(count_abs_errors, dtype=np.float64)

    count_signed_errors = np.asarray(count_signed_errors, dtype=np.float64)

    inference_times_ms = np.asarray(inference_times_ms, dtype=np.float64)


    rows = []

    rows.extend([

        ("Test samples", len(pairs), "Number of image-mask pairs evaluated."),

        ("Fixed ion sites", num_ions, "Ion sites extracted from all-bright GT mask and shared by all frames."),

        ("Model params", total_params / 1e6, "Total number of model parameters in millions."),

        ("Trainable params", trainable_params / 1e6, "Trainable parameters in millions; equal to total at evaluation."),

        ("Mean true bright count", float(np.mean(true_counts)), "Average number of bright ions per test frame from site labels."),

        ("Mask Dice", mask_mean["dice"], "Mean pixel-level Dice between predicted mask and GT mask."),

        ("Mask IoU", mask_mean["iou"], "Mean pixel-level intersection-over-union."),

        ("Mask Precision", mask_mean["precision"], "Mean pixel precision for foreground ion pixels."),

        ("Mask Recall", mask_mean["recall"], "Mean pixel recall for foreground ion pixels."),

        ("Mask F1", mask_mean["f1"], "Mean pixel F1 score."),

        ("Mask Pixel Acc.", mask_mean["accuracy"], "Mean pixel accuracy including background."),

        ("Count MAE", float(count_abs_errors.mean()), "Mean absolute error of bright-ion count per frame."),

        ("Count signed error", float(count_signed_errors.mean()), "Mean predicted minus true bright-ion count."),

        ("Exact count match", float(np.mean(exact_count_matches)), "Fraction of frames with exactly correct bright-ion count."),

        ("State-count fidelity", 1.0 - float(count_abs_errors.sum()) / max(1.0, float(np.sum(true_counts))), "1 - total absolute count error / total true bright count."),

        ("Latency mean ms", float(inference_times_ms.mean()), "Mean single-frame model inference latency."),

        ("Latency median ms", float(np.median(inference_times_ms)), "Median single-frame inference latency."),

        ("Latency p95 ms", float(np.percentile(inference_times_ms, 95)), "95th percentile inference latency."),

        ("Throughput FPS", 1000.0 / float(inference_times_ms.mean()), "Approximate FPS from mean latency."),

    ])


    if all_state_true:

        y_true = np.concatenate(all_state_true)

        y_prob = np.concatenate(all_state_prob)

        y_pred = np.concatenate(all_state_pred)

        s = calc_binary_metrics(y_pred, y_true)

        brier = float(np.mean((y_prob - y_true) ** 2))

        nll = safe_nll(y_true, y_prob)

        ece = expected_calibration_error(y_true, y_prob, n_bins=args.ece_bins)

        auroc = binary_auroc(y_true, y_prob)

        auprc = binary_average_precision(y_true, y_prob)

        dark_true = 1 - y_true

        dark_pred = 1 - y_pred

        dark = calc_binary_metrics(dark_pred, dark_true)

        coord_l2_all = np.concatenate(all_coord_l2)

        coord_l1_all = np.concatenate(all_coord_l1)

        exist_prob_all = np.concatenate(all_exist_prob) if all_exist_prob else None

        uncertainty_all = np.concatenate(all_uncertainty) if all_uncertainty else None


        rows.extend([

            ("Per-ion Acc.", s["accuracy"], "Accuracy over all ion-site bright/dark labels."),

            ("Bright Precision", s["precision"], "Precision for bright-state ion prediction."),

            ("Bright Recall", s["recall"], "Recall for bright-state ions."),

            ("Bright F1", s["f1"], "F1 for bright-state ions."),

            ("Dark Recall", dark["recall"], "Recall for dark-state ions; important under neighbor crosstalk."),

            ("Balanced Acc.", 0.5 * (s["recall"] + s["specificity"]), "Average of bright recall and dark recall."),

            ("Bitstring exact match", float(np.mean(bitstring_matches)), "Fraction of frames with all ion-site states correct."),

            ("AUROC", auroc, "Threshold-free bright/dark ranking metric."),

            ("AUPRC", auprc, "Average precision for bright-state predictions."),

            ("Brier score", brier, "Mean squared error of bright probabilities; lower is better."),

            ("NLL", nll, "Binary negative log-likelihood; lower is better."),

            ("ECE", ece, f"Expected calibration error with {args.ece_bins} bins; lower is better."),

            ("Coord L2 mean px", float(coord_l2_all.mean()), "Mean Euclidean distance between predicted and fixed site coordinates."),

            ("Coord L2 median px", float(np.median(coord_l2_all)), "Median coordinate refinement error."),

            ("Coord L2 p95 px", float(np.percentile(coord_l2_all, 95)), "95th percentile coordinate refinement error."),

            ("Coord L1 mean px", float(coord_l1_all.mean()), "Mean L1 coordinate refinement error."),

        ])

        if exist_prob_all is not None:

            rows.append(("Exist prob mean", float(exist_prob_all.mean()), "Mean predicted existence probability; target is 1 for fixed 300-ion data."))

            rows.append(("Exist Acc.", float(np.mean(exist_prob_all >= 0.5)), "Existence accuracy with all fixed sites treated as present."))

        if uncertainty_all is not None:

            rows.append(("Uncertainty mean", float(uncertainty_all.mean()), "Mean model-predicted readout uncertainty."))

            rows.append(("Uncertainty p95", float(np.percentile(uncertainty_all, 95)), "95th percentile predicted readout uncertainty."))


    table = make_table(rows)


    report_path = Path(args.result_dir) / args.report_name

    per_sample_path = Path(args.result_dir) / "per_sample_metrics.csv"

    with open(per_sample_path, "w", encoding="utf-8") as f:

        f.write("index,image_name,true_count,pred_count,count_error,mask_dice,mask_iou,inference_ms\n")

        f.write("\n".join(per_sample_lines))

        f.write("\n")


    missing = getattr(incompatible, "missing_keys", []) if incompatible is not None else []

    unexpected = getattr(incompatible, "unexpected_keys", []) if incompatible is not None else []

    explanation = f"""
Evaluation summary
==================
Model checkpoint : {args.model_path}
Model arch       : {args.model_arch}
Test root        : {args.test_root}
GT site mask     : {args.gt_all_bright_mask}
Result directory : {args.result_dir}
Per-sample CSV   : {per_sample_path}
Load missing keys: {len(missing)}
Load unexpected  : {len(unexpected)}

{table}

Brief explanation
-----------------
    This report combines segmentation metrics, structured per-ion readout metrics,
    counting metrics, calibration metrics, localization metrics, and latency statistics.
    For mask-based benchmark folders, the test-reference state for each calibrated site is
    read from the corresponding evaluation mask using the configured threshold
    ({MASK_THRESHOLD}).
"""

    with open(report_path, "w", encoding="utf-8") as f:

        f.write(explanation)


    print("\nEvaluation finished.")

    print(f"Report: {report_path}")

    print(f"Per-sample CSV: {per_sample_path}")

    print(table)


def parse_args():

    parser = argparse.ArgumentParser(description="Evaluate DW-UNet / Site-DIA on intersection_test_data.")

    parser.add_argument("--model_path", type=str, default=DEFAULT_MODEL_PATH)

    parser.add_argument("--model_arch", type=str, default="site_dia", choices=["site_dia", "detr_query", "set_transformer", "dwunet", "standard_unet", "vit_unet", "setr", "segformer", "segmenter"])

    parser.add_argument("--test_root", type=str, default=DEFAULT_TEST_ROOT)

    parser.add_argument("--gt_all_bright_mask", type=str, default=DEFAULT_GT_ALL_BRIGHT_MASK)

    parser.add_argument("--result_dir", type=str, default=DEFAULT_RESULT_DIR)

    parser.add_argument("--report_name", type=str, default="neurips_metrics_report.txt")

    parser.add_argument("--mask_threshold", type=float, default=PRED_THRESHOLD)

    parser.add_argument("--state_threshold", type=float, default=PRED_THRESHOLD)

    parser.add_argument("--ece_bins", type=int, default=15)

    parser.add_argument("--num_ion_attn_layers", type=int, default=1)

    parser.add_argument("--allow_partial_load", action="store_true", help="Allow missing/unexpected checkpoint keys.")

    parser.add_argument("--save_pred_masks", action="store_true")

    parser.add_argument("--log_every", type=int, default=100)

    return parser.parse_args()


def main():

    args = parse_args()

    start = time.time()

    evaluate(args)

    print(f"Total runtime: {time.time() - start:.2f}s")


if __name__ == "__main__":

    main()

