import argparse

import csv

import os

from dataclasses import dataclass

from typing import Callable, Dict


import cv2

import matplotlib.pyplot as plt

import numpy as np


@dataclass

class MaskResult:

    name: str

    mask: np.ndarray

    threshold_desc: str


def normalize_for_display(

    image: np.ndarray, lower_q: float = 1.0, upper_q: float = 99.5

) -> np.ndarray:

    lo = np.percentile(image, lower_q)

    hi = np.percentile(image, upper_q)

    scaled = np.clip((image - lo) / max(hi - lo, 1e-8), 0.0, 1.0)

    return scaled


def cleanup_mask(

    mask: np.ndarray, min_area: int = 4, max_area: int = 500

) -> np.ndarray:

    mask_u8 = (mask > 0).astype(np.uint8)

    kernel = np.ones((3, 3), np.uint8)

    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, kernel, iterations=1)

    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kernel, iterations=1)


    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(

        mask_u8, connectivity=8

    )

    filtered = np.zeros_like(mask_u8)

    for label_id in range(1, num_labels):

        area = stats[label_id, cv2.CC_STAT_AREA]

        if min_area <= area <= max_area:

            filtered[labels == label_id] = 1

    return filtered


def mask_otsu_blur(img8: np.ndarray) -> MaskResult:

    blur = cv2.GaussianBlur(img8, (5, 5), 0)

    thr, mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return MaskResult(

        "otsu_blur", cleanup_mask(mask), f"Otsu threshold on blurred image: {thr:.2f}"

    )


def mask_adaptive_gaussian(img8: np.ndarray) -> MaskResult:

    blur = cv2.GaussianBlur(img8, (5, 5), 0)

    mask = cv2.adaptiveThreshold(

        blur,

        255,

        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,

        cv2.THRESH_BINARY,

        21,

        -3,

    )

    return MaskResult(

        "adaptive_gaussian",

        cleanup_mask(mask),

        "Adaptive Gaussian threshold, block=21, C=-3",

    )


def mask_percentile_high(norm: np.ndarray) -> MaskResult:

    blur = cv2.GaussianBlur(norm.astype(np.float32), (5, 5), 0)

    thr = float(np.percentile(blur, 99.2))

    mask = (blur >= thr).astype(np.uint8)

    return MaskResult(

        "percentile_99_2",

        cleanup_mask(mask),

        f"Threshold at 99.2 percentile: {thr:.4f}",

    )


def mask_tophat_otsu(img8: np.ndarray) -> MaskResult:

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))

    tophat = cv2.morphologyEx(img8, cv2.MORPH_TOPHAT, kernel)

    blur = cv2.GaussianBlur(tophat, (5, 5), 0)

    thr, mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return MaskResult(

        "tophat_otsu", cleanup_mask(mask), f"Top-hat + Otsu threshold: {thr:.2f}"

    )


def component_stats(mask: np.ndarray) -> Dict[str, float]:

    mask_u8 = (mask > 0).astype(np.uint8)

    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)

    areas = (

        stats[1:, cv2.CC_STAT_AREA] if num_labels > 1 else np.array([], dtype=np.int32)

    )

    return {

        "foreground_pixels": int(mask_u8.sum()),

        "components": int(len(areas)),

        "mean_component_area": float(areas.mean()) if len(areas) else 0.0,

        "max_component_area": int(areas.max()) if len(areas) else 0,

    }


def save_overlay(

    base_rgb: np.ndarray, mask: np.ndarray, title: str, out_path: str

) -> None:

    overlay = base_rgb.copy()

    overlay[mask > 0] = np.array([255, 64, 64], dtype=np.uint8)

    blended = cv2.addWeighted(base_rgb, 0.7, overlay, 0.3, 0)

    plt.figure(figsize=(10, 3))

    plt.imshow(blended)

    plt.title(title)

    plt.axis("off")

    plt.tight_layout()

    plt.savefig(out_path, dpi=200, bbox_inches="tight")

    plt.close()


def main() -> None:

    parser = argparse.ArgumentParser(

        description="Generate multiple mask proposals from averaged bright-frame data."

    )

    parser.add_argument("--input", required=True, help="Path to bright.npy")

    parser.add_argument(

        "--output_dir",

        default="mask_proposals",

        help="Directory to save generated masks and previews",

    )

    args = parser.parse_args()


    os.makedirs(args.output_dir, exist_ok=True)


    stack = np.load(args.input)

    if stack.ndim != 3:

        raise ValueError(

            f"Expected a 3D array shaped like (frames, height, width), got {stack.shape}"

        )


    avg = stack.mean(axis=0).astype(np.float32)

    norm = normalize_for_display(avg)

    img8 = np.round(norm * 255).astype(np.uint8)

    avg_rgb = np.dstack([img8, img8, img8])


    cv2.imwrite(os.path.join(args.output_dir, "average_image.png"), img8)


    methods: Dict[str, Callable[[], MaskResult]] = {

        "otsu_blur": lambda: mask_otsu_blur(img8),

        "adaptive_gaussian": lambda: mask_adaptive_gaussian(img8),

        "percentile_99_2": lambda: mask_percentile_high(norm),

        "tophat_otsu": lambda: mask_tophat_otsu(img8),

    }


    rows = []

    preview_items = []


    for name, fn in methods.items():

        result = fn()

        mask_u8 = (result.mask * 255).astype(np.uint8)

        stats = component_stats(result.mask)


        mask_path = os.path.join(args.output_dir, f"{name}_mask.png")

        overlay_path = os.path.join(args.output_dir, f"{name}_overlay.png")

        cv2.imwrite(mask_path, mask_u8)

        save_overlay(avg_rgb, result.mask, f"{name} overlay", overlay_path)


        rows.append(

            {

                "method": result.name,

                "threshold_desc": result.threshold_desc,

                **stats,

            }

        )

        preview_items.append((result.name, mask_u8, stats["components"]))


    with open(

        os.path.join(args.output_dir, "summary.csv"), "w", newline="", encoding="utf-8"

    ) as f:

        writer = csv.DictWriter(

            f,

            fieldnames=[

                "method",

                "threshold_desc",

                "foreground_pixels",

                "components",

                "mean_component_area",

                "max_component_area",

            ],

        )

        writer.writeheader()

        writer.writerows(rows)


    fig, axes = plt.subplots(

        len(preview_items) + 1, 2, figsize=(12, 3 * (len(preview_items) + 1))

    )

    axes[0, 0].imshow(avg, cmap="gray")

    axes[0, 0].set_title("Average image (raw scale)")

    axes[0, 0].axis("off")

    axes[0, 1].imshow(img8, cmap="gray")

    axes[0, 1].set_title("Average image (display normalized)")

    axes[0, 1].axis("off")


    for idx, (name, mask_u8, components) in enumerate(preview_items, start=1):

        axes[idx, 0].imshow(img8, cmap="gray")

        axes[idx, 0].set_title(f"{name} source")

        axes[idx, 0].axis("off")

        axes[idx, 1].imshow(mask_u8, cmap="gray")

        axes[idx, 1].set_title(f"{name} mask | components={components}")

        axes[idx, 1].axis("off")


    fig.tight_layout()

    fig.savefig(os.path.join(args.output_dir, "comparison_panel.png"), dpi=200)

    plt.close(fig)


if __name__ == "__main__":

    main()

