import argparse

import json

import logging

import os

import random

from pathlib import Path


import cv2

import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

import torch

from sklearn.model_selection import train_test_split

try:

    from tensorboardX import SummaryWriter

except ImportError:

    class SummaryWriter:  # TensorBoard logging is optional for numerical runs.

        def __init__(self, *args, **kwargs):

            pass

        def add_scalars(self, *args, **kwargs):

            pass

        def close(self):

            pass

from torch.optim.lr_scheduler import ReduceLROnPlateau

from torch.utils.data import DataLoader

from torchvision.transforms import Compose, ToTensor


from dataset import MaskDataset, get_img_files

from loss import (

    HybridSegmentationMultiIonLoss,

    SiteDIAMultitaskLoss,

    PSFForwardModel,

    ReconLossSchedule,

    compute_saturation_threshold,

)

from nets.DWNetV2_unet import DWNetV2_unet

from nets.DETRQuery import DETRStyleQuery

from nets.SetTransformer import SetTransformerReadout

from nets.StandardUNet import StandardUNet

from nets.ViTUNet import ViTUNet

from nets.SETR import SETR

from nets.SegFormer import SegFormer

from nets.Segmenter import Segmenter

from trainer import Trainer


torch.backends.cudnn.deterministic = True

torch.backends.cudnn.benchmark = False


DEFAULT_PRETRAIN_CKPT = os.environ.get("PRETRAIN_CKPT", "")


BATCH_SIZE = 48

NUM_WORKERS = 2

INITIAL_LR = 1e-4

RANDOM_STATE = 1

NUM_EPOCHS = 100

VAL_RATIO = 0.2

SAMPLE_SIZES = [50000]


EXPERIMENT = "pretrained_random_subsets_single_run"

OUT_DIR = Path("outputs") / EXPERIMENT


def seed_worker(worker_id):

    worker_seed = torch.initial_seed() % 2**32

    np.random.seed(worker_seed)

    random.seed(worker_seed)


def get_data_loaders(

    train_files,

    val_files,

    batch_size,

    site_dia_label_dir=None,

    require_mask=True,

):

    train_transform = Compose(

        [

            ToTensor(),

        ]

    )


    val_transform = Compose(

        [

            ToTensor(),

        ]

    )


    g = torch.Generator()

    g.manual_seed(RANDOM_STATE)


    train_loader = DataLoader(

        MaskDataset(

            train_files,

            transform=train_transform,

            mask_transform=val_transform,

            site_dia_label_dir=site_dia_label_dir,

            require_mask=require_mask,

        ),

        batch_size=batch_size,

        shuffle=True,

        pin_memory=True,

        num_workers=NUM_WORKERS,

        worker_init_fn=seed_worker,

        generator=g,

    )


    val_loader = DataLoader(

        MaskDataset(

            val_files,

            transform=val_transform,

            mask_transform=val_transform,

            site_dia_label_dir=site_dia_label_dir,

            require_mask=require_mask,

        ),

        batch_size=batch_size,

        shuffle=False,

        pin_memory=True,

        num_workers=NUM_WORKERS,

        worker_init_fn=seed_worker,

    )


    return train_loader, val_loader


def build_logger(log_path):

    logger = logging.getLogger("train_subset_logger")

    logger.setLevel(logging.DEBUG)

    logger.handlers.clear()

    logger.addHandler(logging.FileHandler(filename=log_path, encoding="utf-8"))

    logger.addHandler(logging.StreamHandler())

    return logger


def save_best_model(output_dir, model, df_hist, metric="val_loss"):

    current = df_hist[metric].tail(1).iloc[0]

    is_best = current >= df_hist[metric].max() if metric == "val_ion_acc" else current <= df_hist[metric].min()

    if is_best:

        torch.save(model.state_dict(), output_dir / "best.pth")


def write_on_board(writer, experiment_name, df_hist):

    row = df_hist.tail(1).iloc[0]


    writer.add_scalars(

        f"{experiment_name}/loss",

        {

            "train": row.train_loss,

            "val": row.val_loss,

        },

        row.epoch,

    )


def log_hist(logger, df_hist, metric="val_loss"):

    last = df_hist.tail(1)

    best = df_hist.sort_values(metric, ascending=metric != "val_ion_acc").head(1)

    summary = pd.concat((last, best)).reset_index(drop=True)

    summary["name"] = ["Last", "Best"]

    fields = ["name", "epoch", "train_loss", "val_loss", "current_lr"]

    if metric == "val_ion_acc":

        fields.extend(["val_ion_acc", "decoder_radius", "decoder_threshold"])

    logger.debug(summary[fields])

    logger.debug("")


def save_loss_plot(df_hist, output_path, plot_title):

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(df_hist["epoch"], df_hist["train_loss"], label="Train Loss", linewidth=2)

    ax.plot(df_hist["epoch"], df_hist["val_loss"], label="Val Loss", linewidth=2)

    ax.set_xlabel("Epoch")

    ax.set_ylabel("Loss")

    ax.set_title(plot_title)

    ax.grid(True, linestyle="--", alpha=0.4)

    ax.legend()

    fig.tight_layout()

    fig.savefig(output_path, dpi=200)

    plt.close(fig)


def load_pretrained_weights(model, ckpt_path, device, load_mode="full"):

    if not ckpt_path:

        print("No pretrained checkpoint provided.")

        return model


    if not os.path.exists(ckpt_path):

        raise FileNotFoundError(f"Pretrained checkpoint not found: {ckpt_path}")


    ckpt = torch.load(ckpt_path, map_location=device)


    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:

        state_dict = ckpt["model_state_dict"]

    elif isinstance(ckpt, dict) and "state_dict" in ckpt:

        state_dict = ckpt["state_dict"]

    else:

        state_dict = ckpt


    stripped = {}

    for k, v in state_dict.items():

        new_k = k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k

        stripped[new_k] = v

    state_dict = stripped


    model_state = model.state_dict()

    loaded_keys = []


    if load_mode == "full":

        filtered = {}

        for k, v in state_dict.items():

            if k in model_state and model_state[k].shape == v.shape:

                filtered[k] = v

                loaded_keys.append(k)


        incompatible = model.load_state_dict(filtered, strict=False)

        print(f"[Pretrain-FULL] loaded={len(loaded_keys)}")

        print(f"[Pretrain-FULL] missing={len(incompatible.missing_keys)}")

        print(f"[Pretrain-FULL] unexpected={len(incompatible.unexpected_keys)}")


    elif load_mode == "backbone":

        filtered = {}

        for k, v in state_dict.items():

            candidate_keys = [k]


            if k.startswith("backbone."):

                candidate_keys.append(k[len("backbone."):])


            for ck in candidate_keys:

                if ck in model_state and model_state[ck].shape == v.shape:

                    filtered[ck] = v

                    loaded_keys.append(ck)

                    break


                bk = f"backbone.{ck}"

                if bk in model_state and model_state[bk].shape == v.shape:

                    filtered[bk] = v

                    loaded_keys.append(bk)

                    break


        incompatible = model.load_state_dict(filtered, strict=False)

        print(f"[Pretrain-BACKBONE] loaded={len(loaded_keys)}")

        print(f"[Pretrain-BACKBONE] missing={len(incompatible.missing_keys)}")

        print(f"[Pretrain-BACKBONE] unexpected={len(incompatible.unexpected_keys)}")


    else:

        raise ValueError(f"Unsupported load_mode: {load_mode}")


    return model


def select_subset(image_files, sample_size, seed):

    if sample_size > len(image_files):

        raise ValueError(

            f"Requested sample size {sample_size}, but only {len(image_files)} images are available."

        )


    rng = np.random.default_rng(seed)

    selected_indices = np.sort(

        rng.choice(len(image_files), size=sample_size, replace=False)

    )

    return image_files[selected_indices]


def split_train_val(subset_files, val_ratio, seed):

    train_files, val_files = train_test_split(

        subset_files,

        test_size=val_ratio,

        random_state=seed,

        shuffle=True,

    )

    return np.array(train_files), np.array(val_files)


def save_split_manifest(train_files, val_files, output_path):

    split_df = pd.DataFrame(

        {

            "file": list(train_files) + list(val_files),

            "split": ["train"] * len(train_files) + ["val"] * len(val_files),

        }

    )

    split_df.to_csv(output_path, index=False)


def write_experiment_readme(

    output_dir, dataset_dir, pretrained_ckpt, load_mode, batch_size, use_amp

):

    readme_text = f"""Pretrained Training on Random Subsets

This directory stores pretrained training runs on random subsets of the dataset.

Dataset

- Dataset root: `{dataset_dir}`
- Image counts tested: {", ".join(str(size) for size in SAMPLE_SIZES)}
- Validation split ratio: {VAL_RATIO}
- Max epochs per run: {NUM_EPOCHS}
- Batch size: {batch_size}
- AMP mixed precision: {use_amp}

Pretrained Model

- Checkpoint: `{pretrained_ckpt}`
- Load mode: `{load_mode}`

Output Structure

Each `sample_xxxx` directory contains:

- `best.pth` for the best validation-loss checkpoint
- `hist.csv` with per-epoch losses and learning rate
- `loss.png` with the train/validation loss curve
- `split_manifest.csv` listing the train/validation files
- `tensorboard` logs
- `summary.csv` with the key results for that sample-size run

Run Command

```bash
python train_main.py --pretrained_ckpt "{pretrained_ckpt}" --load_mode {load_mode}
```
"""

    (output_dir / "README.md").write_text(readme_text, encoding="utf-8")


def estimate_saturation_threshold(image_paths, sample_size=500, seed=RANDOM_STATE):

    """Estimates T_sat = Q_0.999 of training-pixel intensities (Site-AIT paper,

    Sec. IV.G) from a random subsample of the training split, once, before

    training starts. The result is frozen for the entire run (not recomputed

    per epoch/batch), matching the paper's description.

    """

    rng = random.Random(seed)

    paths = list(image_paths)

    if len(paths) > sample_size:

        paths = rng.sample(paths, sample_size)

    pixel_arrays = []

    for p in paths:

        img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)

        if img is None:

            continue

        pixel_arrays.append(img.astype("float32") / 255.0)

    if not pixel_arrays:

        raise RuntimeError("No readable training images to estimate the saturation threshold T_sat.")

    return compute_saturation_threshold(pixel_arrays)


def extract_calibrated_sites(mask_path):

    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

    if mask is None:

        raise FileNotFoundError(f"Could not read dense calibration mask: {mask_path}")

    _, binary = cv2.threshold(mask, 128, 255, cv2.THRESH_BINARY)

    count, _, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)

    sites = []

    for component in range(1, count):

        if stats[component, cv2.CC_STAT_AREA] < 2:

            continue

        x, y = centroids[component]

        sites.append((float(x), float(y)))

    sites.sort(key=lambda xy: (xy[1], xy[0]))

    if not sites:

        raise ValueError(f"No calibrated sites found in {mask_path}")

    return np.asarray(sites, dtype=np.float32)


def estimate_positive_class_weight(train_files):

    positives = 0

    pixels = 0

    for image_path in train_files:

        image_path = Path(image_path)

        parts = list(image_path.parts)

        try:

            parts[parts.index("images")] = "masks"

        except ValueError as exc:

            raise ValueError(f"Training image path has no images component: {image_path}") from exc

        mask_path = Path(*parts)

        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

        if mask is None:

            raise FileNotFoundError(f"Could not read training mask: {mask_path}")

        positives += int((mask > 128).sum())

        pixels += int(mask.size)

    negatives = pixels - positives

    if positives == 0:

        raise ValueError("Cannot compute positive-class weight: no foreground pixels")

    return negatives / positives


def run_experiments(

    pre_trained,

    pretrained_ckpt="",

    load_mode="full",

    batch_size=BATCH_SIZE,

    use_amp=True,

    centroid_weight=0.0,

    bce_weight=1.0,

    dice_weight=1.0,

    model_arch="dwunet",

    output_dir=OUT_DIR,

    site_dia_label_dir="",

    site_mask_weight=0.0,

    site_state_weight=1.0,

    site_coord_weight=0.05,

    site_exist_weight=0.1,

    site_offset_reg_weight=0.01,

    freeze_dwunet=False,

    dia_num_ion_attn_layers=1,

    dia_token_dim=256,

    dia_num_heads=8,

    site_use_global_cross_attn=False,

    site_disable_tok_offset=False,

    site_disable_psf_mask=False,

    site_disable_coord_head=False,

    site_tok_offset_max_px=3.0,

    site_disable_recon_loss=False,

    site_recon_weight_max=0.05,

    site_recon_warm_epochs=10,

    site_recon_ramp_epochs=5,

    site_recon_no_warmup=False,

    site_freeze_psf_sigma=False,

    site_psf_sigma_init=1.5,

    site_psf_kernel="isotropic",

    site_psf_beta_init=2.0,

    site_psf_profile="",

    site_recon_lambda_ssim=0.2,

    early_stopping_patience=20,

    early_stopping_metric="val_loss",

    learning_rate=INITIAL_LR,

    dense_calibration_mask="",

    auto_pos_weight=False,

    dense_base_channels=32,

    dense_vit_depth=6,

    dense_patch_size=16,

    dense_setr_decoder="pup",

    dense_segmenter_depth=12,

    detr_token_dim=256,

    detr_num_layers=4,

    detr_num_heads=8,

    detr_template_size=13,

):

    site_token_archs = ("site_dia", "detr_query", "set_transformer")

    require_mask = not (

        (model_arch == "site_dia" and site_mask_weight == 0)

        or model_arch in ("detr_query", "set_transformer")

    )

    image_files = get_img_files(require_masks=require_mask)

    if len(image_files) == 0:

        raise RuntimeError(

            "No training images found. Please check IMG_DIR/images and IMG_DIR/masks."

        )


    output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    logger = build_logger(output_dir / "run.log")

    dense_site_xy = None

    if dense_calibration_mask:

        dense_site_xy = extract_calibrated_sites(dense_calibration_mask)

        logger.info(f"Loaded {len(dense_site_xy)} calibrated sites for validation decoding.")

    dense_archs = ("dwunet", "standard_unet", "vit_unet", "setr", "segformer", "segmenter")

    if (

        early_stopping_metric == "val_ion_acc"

        and model_arch in dense_archs

        and dense_site_xy is None

    ):

        raise ValueError("val_ion_acc early stopping requires --dense_calibration_mask")


    device = torch.device("cuda")

    logger.info(f"GPU: {torch.cuda.get_device_name(0)}")

    logger.info(f"CUDA: {torch.version.cuda}")


    write_experiment_readme(

        output_dir=output_dir,

        dataset_dir=os.environ.get(

            "IMG_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "intersection_train_data")

        ),

        pretrained_ckpt=pretrained_ckpt,

        load_mode=load_mode,

        batch_size=batch_size,

        use_amp=bool(use_amp),

    )


    logger.info(f"Found {len(image_files)} available image/mask pairs.")

    logger.info(f"Model architecture: {model_arch}")

    logger.info(f"Batch size: {batch_size}")

    logger.info(f"AMP mixed precision: {bool(use_amp)}")

    all_summaries = []


    for sample_size in SAMPLE_SIZES:

        logger.info("=" * 80)

        logger.info(f"Starting sample-size experiment: {sample_size}")


        sample_dir = output_dir / f"sample_{sample_size}"

        sample_dir.mkdir(parents=True, exist_ok=True)


        subset_files = select_subset(

            image_files, sample_size, seed=RANDOM_STATE + sample_size

        )

        train_files, val_files = split_train_val(

            subset_files,

            val_ratio=VAL_RATIO,

            seed=RANDOM_STATE + sample_size,

        )

        save_split_manifest(train_files, val_files, sample_dir / "split_manifest.csv")


        writer = SummaryWriter(log_dir=str(sample_dir / "tensorboard"))

        experiment_name = f"{EXPERIMENT}/sample_{sample_size}"


        def on_after_epoch(

            m,

            df_hist,

            current_sample_dir=sample_dir,

            current_experiment_name=experiment_name,

        ):

            save_best_model(current_sample_dir, m, df_hist, metric=early_stopping_metric)

            write_on_board(writer, current_experiment_name, df_hist)

            log_hist(logger, df_hist, metric=early_stopping_metric)


        if model_arch in site_token_archs:

            if not site_dia_label_dir:

                site_dia_label_dir = os.environ.get("SITE_DIA_LABEL_DIR", "")

            if not site_dia_label_dir:

                raise ValueError(

                    f"{model_arch} requires --site_dia_label_dir or SITE_DIA_LABEL_DIR."

                )


        if model_arch == "detr_query" or model_arch == "set_transformer":

            # App. F: the two coordinate-aware baselines are only described with

            # a bright-state classification head (no coordinate refinement,

            # existence prediction, dense mask, or reconstruction consistency),

            # so they train under a reduced multitask loss: state BCE only.

            recon_schedule = None

            criterion = SiteDIAMultitaskLoss(

                mask_weight=0.0,

                state_weight=site_state_weight,

                coord_weight=0.0,

                exist_weight=0.0,

                offset_reg_weight=0.0,

                bce_weight=bce_weight,

                dice_weight=dice_weight,

                psf_model=None,

                t_sat=None,

            )

        elif model_arch == "site_dia":

            if site_disable_recon_loss:

                psf_model = None

                t_sat = None

                recon_schedule = None

            else:

                init_profile = None

                if site_psf_profile:

                    init_profile = np.load(site_psf_profile)

                    logger.info(

                        f"Loaded measured PSF profile {init_profile.shape} from {site_psf_profile}"

                    )

                psf_model = PSFForwardModel(

                    sigma_init=site_psf_sigma_init,

                    recon_radius=6,

                    kernel=site_psf_kernel,

                    beta_init=site_psf_beta_init,

                    init_profile=init_profile,

                )

                logger.info(f"PSF forward model kernel: {site_psf_kernel}")

                if site_freeze_psf_sigma:

                    psf_model.freeze_shape()

                t_sat = estimate_saturation_threshold(train_files)

                logger.info(f"Estimated saturation threshold T_sat = {t_sat:.6f}")

                recon_schedule = ReconLossSchedule(

                    warm_epochs=site_recon_warm_epochs,

                    ramp_epochs=site_recon_ramp_epochs,

                    max_weight=site_recon_weight_max,

                    use_warmup=not site_recon_no_warmup,

                    freeze_sigma=site_freeze_psf_sigma,

                    disabled=False,

                )

            criterion = SiteDIAMultitaskLoss(

                mask_weight=site_mask_weight,

                state_weight=site_state_weight,

                coord_weight=site_coord_weight,

                exist_weight=site_exist_weight,

                offset_reg_weight=site_offset_reg_weight,

                bce_weight=bce_weight,

                dice_weight=dice_weight,

                psf_model=psf_model,

                t_sat=t_sat,

                recon_lambda_ssim=site_recon_lambda_ssim,

            )

        else:

            recon_schedule = None

            positive_class_weight = None

            if auto_pos_weight:

                positive_class_weight = estimate_positive_class_weight(train_files)

                logger.info(

                    f"Training-only positive-class weight: {positive_class_weight:.6f}"

                )

            criterion = HybridSegmentationMultiIonLoss(

                bce_weight=bce_weight,

                dice_weight=dice_weight,

                centroid_weight=centroid_weight,

                radius=4,

                positive_class_weight=positive_class_weight,

            )


        data_loaders = get_data_loaders(

            train_files,

            val_files,

            batch_size=batch_size,

            site_dia_label_dir=site_dia_label_dir if model_arch in site_token_archs else None,

            require_mask=require_mask,

        )


        if model_arch == "dwunet":

            model = DWNetV2_unet(pre_trained)

        elif model_arch == "site_dia":

            model = DWNetV2_unet(

                pre_trained,

                enable_site_dia=True,

                num_ions=300,

                dia_token_dim=dia_token_dim,

                dia_num_heads=dia_num_heads,

                dia_num_ion_attn_layers=dia_num_ion_attn_layers,

                dia_tok_offset_max_px=site_tok_offset_max_px,

                dia_use_global_cross_attn=site_use_global_cross_attn,

                dia_disable_offset=site_disable_tok_offset,

                dia_disable_psf_mask=site_disable_psf_mask,

                dia_disable_coord_head=site_disable_coord_head,

            )

        elif model_arch == "detr_query":

            if detr_template_size % 2 != 1:

                raise ValueError("DETR template size must be odd")

            model = DETRStyleQuery(

                num_ions=300,

                token_dim=detr_token_dim,

                num_heads=detr_num_heads,

                num_layers=detr_num_layers,

                window_radius=(detr_template_size - 1) // 2,

            )

        elif model_arch == "set_transformer":

            model = SetTransformerReadout(num_ions=300)

        elif model_arch == "standard_unet":

            model = StandardUNet(base_channels=dense_base_channels)

            if pretrained_ckpt:

                logger.warning(

                    "standard_unet does not use DW-UNet pretraining; training from scratch."

                )

                pretrained_ckpt = ""

        elif model_arch == "vit_unet":

            model = ViTUNet(

                base_channels=dense_base_channels,

                vit_depth=dense_vit_depth,

            )

            if pretrained_ckpt:

                logger.warning("vit_unet does not use DW-UNet pretraining; training from scratch.")

                pretrained_ckpt = ""

        elif model_arch == "setr":

            model = SETR(

                patch_size=dense_patch_size,

                decoder=dense_setr_decoder,

            )

            if pretrained_ckpt:

                logger.warning("setr does not use DW-UNet pretraining; training from scratch.")

                pretrained_ckpt = ""

        elif model_arch == "segformer":

            model = SegFormer()

            if pretrained_ckpt:

                logger.warning("segformer does not use DW-UNet pretraining; training from scratch.")

                pretrained_ckpt = ""

        elif model_arch == "segmenter":

            model = Segmenter(

                patch_size=dense_patch_size,

                encoder_depth=dense_segmenter_depth,

            )

            if pretrained_ckpt:

                logger.warning("segmenter does not use DW-UNet pretraining; training from scratch.")

                pretrained_ckpt = ""

        else:

            raise ValueError(f"Unsupported model_arch: {model_arch}")


        if model_arch == "site_dia" and freeze_dwunet:

            for name, param in model.named_parameters():

                param.requires_grad = name.startswith("site_dia.")

            logger.info("Frozen DW-UNet backbone/decoder; training Site-DIA head only.")


        model.to(device)

        criterion.to(device)


        if pretrained_ckpt:

            logger.info(f"Loading pretrained checkpoint: {pretrained_ckpt}")

            logger.info(f"Load mode: {load_mode}")

            model = load_pretrained_weights(

                model=model,

                ckpt_path=pretrained_ckpt,

                device=device,

                load_mode=load_mode,

            )


        trainable_params = [p for p in model.parameters() if p.requires_grad]

        trainable_params += [p for p in criterion.parameters() if p.requires_grad]

        optimizer = torch.optim.Adam(

            trainable_params, lr=learning_rate, weight_decay=1e-5

        )


        scheduler = ReduceLROnPlateau(

            optimizer,

            mode="min",

            factor=0.75,

            patience=3,

            threshold=1e-4,

            min_lr=1e-6,

        )


        trainer = Trainer(

            data_loaders=data_loaders,

            criterion=criterion,

            device=device,

            scheduler=scheduler,

            on_after_epoch=on_after_epoch,

            use_amp=use_amp,

            recon_schedule=recon_schedule,

            early_stopping_patience=early_stopping_patience,

            early_stopping_metric=early_stopping_metric,

            dense_site_xy=dense_site_xy,

        )


        hist = trainer.train(model, optimizer, num_epochs=NUM_EPOCHS)

        hist.to_csv(sample_dir / "hist.csv", index=False)

        save_loss_plot(

            hist,

            output_path=sample_dir / "loss.png",

            plot_title=f"Sample {sample_size} Loss Curve",

        )


        best_index = (

            hist["val_ion_acc"].idxmax()

            if early_stopping_metric == "val_ion_acc"

            else hist["val_loss"].idxmin()

        )

        best_row = hist.loc[best_index]

        last_row = hist.iloc[-1]

        summary_df = pd.DataFrame(

            [

                {

                    "sample_size": sample_size,

                    "num_epochs": NUM_EPOCHS,

                    "train_size": len(train_files),

                    "val_size": len(val_files),

                    "best_epoch": int(best_row["epoch"]),

                    "best_val_loss": float(best_row["val_loss"]),

                    "last_val_loss": float(last_row["val_loss"]),

                    "last_lr": float(last_row["current_lr"]),

                    "best_val_ion_acc": float(best_row["val_ion_acc"]),

                    "decoder_radius": float(best_row["decoder_radius"]),

                    "decoder_threshold": float(best_row["decoder_threshold"]),

                    "decoder_youden_j": float(best_row["decoder_youden_j"]),

                }

            ]

        )

        summary_df.to_csv(sample_dir / "summary.csv", index=False)

        all_summaries.append(summary_df.iloc[0].to_dict())

        if early_stopping_metric == "val_ion_acc":

            candidate_result = {

                "val_ion_acc": float(best_row["val_ion_acc"]),

                "best_epoch": int(best_row["epoch"]),

                "checkpoint": str(sample_dir / "best.pth"),

            }

            if np.isfinite(best_row["decoder_radius"]):

                candidate_result.update(

                    decoder_radius=int(best_row["decoder_radius"]),

                    decoder_threshold=float(best_row["decoder_threshold"]),

                    decoder_youden_j=float(best_row["decoder_youden_j"]),

                )

            (output_dir / "candidate_result.json").write_text(

                json.dumps(candidate_result, indent=2), encoding="utf-8"

            )


        writer.close()

        del (

            trainer,

            optimizer,

            scheduler,

            model,

            data_loaders,

            criterion,

            hist,

            summary_df,

        )

        if torch.cuda.is_available():

            torch.cuda.empty_cache()


    pd.DataFrame(all_summaries).to_csv(

        output_dir / "all_results_summary.csv", index=False

    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(

        "--pretrained_ckpt",

        type=str,

        default=DEFAULT_PRETRAIN_CKPT,

        help="Path to pretrained checkpoint",

    )

    parser.add_argument(

        "--load_mode", type=str, default="full", choices=["full", "backbone"]

    )

    parser.add_argument(

        "--batch_size",

        type=int,

        default=BATCH_SIZE,

        help="Batch size for both training and validation",

    )

    parser.add_argument(

        "--disable_amp",

        action="store_true",

        help="Disable CUDA mixed precision training",

    )

    parser.add_argument(

        "--from_scratch",

        action="store_true",

        help="Do not load a self-supervised pretrained checkpoint",

    )

    parser.add_argument(

        "--centroid_weight",

        type=float,

        default=0.0,

        help="Weight for the multi-ion centroid/localization loss ablation",

    )

    parser.add_argument(

        "--bce_weight",

        type=float,

        default=1.0,

        help="Weight for BCEWithLogits loss; set to 0 for Dice-only training",

    )

    parser.add_argument(

        "--dice_weight",

        type=float,

        default=1.0,

        help="Weight for Dice loss",

    )

    parser.add_argument(

        "--sample_sizes",

        type=str,

        default=",".join(str(x) for x in SAMPLE_SIZES),

        help="Comma-separated subset sizes, e.g. 1000,2000,5000",

    )

    parser.add_argument(

        "--epochs",

        type=int,

        default=NUM_EPOCHS,

        help="Number of epochs per subset experiment",

    )

    parser.add_argument(

        "--model_arch",

        type=str,

        default="dwunet",

        choices=["dwunet", "standard_unet", "site_dia", "detr_query", "set_transformer", "vit_unet", "setr", "segformer", "segmenter"],

        help="Architecture to train for architecture ablations",

    )

    parser.add_argument(

        "--output_dir",

        type=str,

        default=str(OUT_DIR),

        help="Output directory for this experiment; use separate dirs for ablations",

    )

    parser.add_argument(

        "--site_dia_label_dir",

        type=str,

        default=os.environ.get("SITE_DIA_LABEL_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "intersection_train_data", "site_dia_labels")),

        help="Directory containing Site-DIA site-level center/state labels",

    )

    parser.add_argument("--site_mask_weight", type=float, default=0.0, help="Site-AIT's inference path never predicts a dense mask (paper Sec. III); leave at 0 unless deliberately training the diagnostic mask head.")

    parser.add_argument("--site_state_weight", type=float, default=1.0)

    parser.add_argument("--site_coord_weight", type=float, default=0.05)

    parser.add_argument("--site_exist_weight", type=float, default=0.1)

    parser.add_argument("--site_offset_reg_weight", type=float, default=0.01)

    parser.add_argument(

        "--freeze_dwunet",

        action="store_true",

        help="For Site-DIA stage-2 training: freeze the DW-UNet backbone/decoder and train only the Site-DIA head.",

    )

    parser.add_argument(

        "--dia_num_ion_attn_layers",

        type=int,

        default=1,

        help="Number of ion-ion self-attention layers for PSF-crosstalk suppression (0 to disable, Table VI 'w/o ion self-attn').",

    )

    parser.add_argument("--dia_token_dim", type=int, default=256, help="Lattice-aware site-token dimension D (paper App. F: D=256).")

    parser.add_argument("--dia_num_heads", type=int, default=8, help="Attention heads for tokenization/ion-interaction (paper Sec. VI A, Implementation details).")

    parser.add_argument(

        "--site_use_global_cross_attn",

        action="store_true",

        help="Table VI 'w/ global cross-attn' ablation: replace adaptive Top-16 tokenization with a generic global cross-attention tokenizer.",

    )

    parser.add_argument(

        "--site_disable_tok_offset",

        action="store_true",

        help="Table VI 'w/o tokenization offsets' ablation: force the tokenization offset Delta^tok to 0.",

    )

    parser.add_argument(

        "--site_disable_psf_mask",

        action="store_true",

        help="Table VI 'w/o learned PSF mask' ablation: use a uniform post-selection weight instead of the learnable PSF-inspired mask.",

    )

    parser.add_argument(

        "--site_disable_coord_head",

        action="store_true",

        help="Table VI 'w/o learn. coord. constraint' ablation: skip the coordinate head, always output the nominal coordinate.",

    )

    parser.add_argument(

        "--site_tok_offset_max_px",

        type=float,

        default=3.0,

        help="R_Delta, the max per-axis tokenization offset in pixels (paper Sec. VI A, Implementation details).",

    )

    parser.add_argument(

        "--site_disable_recon_loss",

        action="store_true",

        help="Table VI 'Full - L_recon, no recon. loss' ablation: disable the PSF reconstruction consistency term entirely.",

    )

    parser.add_argument("--site_recon_weight_max", type=float, default=0.05, help="lambda_recon^max (paper App. H).")

    parser.add_argument("--site_recon_warm_epochs", type=int, default=10, help="E_warm (paper App. H).")

    parser.add_argument("--site_recon_ramp_epochs", type=int, default=5, help="E_ramp (paper App. H).")

    parser.add_argument(

        "--site_recon_no_warmup",

        action="store_true",

        help="Table VI 'Full + L_recon, no warm-up' ablation: use lambda_recon^max from epoch 0, no state-head gradient detachment.",

    )

    parser.add_argument(

        "--site_freeze_psf_sigma",

        action="store_true",

        help="Table VI 'Full + L_recon, sigma_psf frozen' ablation: do not optimize the PSF width.",

    )

    parser.add_argument("--site_psf_sigma_init", type=float, default=1.5, help="Initial sigma_psf in pixels (paper App. H, PSF-consistency implementation).")

    parser.add_argument(

        "--site_psf_kernel",

        type=str,

        default="isotropic",

        choices=["isotropic", "anisotropic", "empirical", "field_dependent"],

        help="PSF profile used by L_recon. 'isotropic' is the published Gaussian. "

             "'anisotropic' is an elliptical Moffat with a directional skew and "

             "'empirical' is a free-form learnable grid; both add the heavy tails "

             "and asymmetry measured in Sec. II.B. 'field_dependent' additionally "

             "lets the profile vary across the register through coma, astigmatism "

             "and micromotion terms whose radial dependence is fixed by aberration "

             "theory (see psf_kernels.py).",

    )

    parser.add_argument("--site_psf_beta_init", type=float, default=2.0, help="Initial Moffat beta for --site_psf_kernel anisotropic. Lower is heavier tailed; beta -> inf is the Gaussian limit.")

    parser.add_argument("--site_psf_profile", type=str, default="", help="Optional .npy with a measured (2R+1, 2R+1) stacked PSF profile used to initialize --site_psf_kernel empirical.")

    parser.add_argument("--site_recon_lambda_ssim", type=float, default=0.2, help="SSIM weight inside L_recon (paper Sec. VI A, Implementation details).")

    parser.add_argument(

        "--random_state",

        type=int,

        default=RANDOM_STATE,

        help="Global random seed for reproducibility (use 1/2/3 for multi-seed runs).",

    )

    parser.add_argument(

        "--early_stopping_patience",

        type=int,

        default=20,

        help="Stop training if val_loss does not improve for this many epochs (default: 20).",

    )

    parser.add_argument(

        "--early_stopping_metric",

        choices=["val_loss", "val_ion_acc"],

        default="val_loss",

        help="Dense search uses validation per-ion accuracy; ordinary runs keep val_loss.",

    )

    parser.add_argument("--learning_rate", type=float, default=INITIAL_LR)

    parser.add_argument(

        "--dense_calibration_mask",

        default="",

        help="All-bright mask defining calibrated sites for nested dense decoder selection.",

    )

    parser.add_argument(

        "--auto_pos_weight",

        action="store_true",

        help="Compute BCE positive-class weight from training masks only.",

    )

    parser.add_argument("--dense_base_channels", type=int, default=32)

    parser.add_argument("--dense_vit_depth", type=int, default=6)

    parser.add_argument("--dense_patch_size", type=int, default=16)

    parser.add_argument(

        "--dense_setr_decoder", choices=["pup", "naive", "mla"], default="pup"

    )

    parser.add_argument("--dense_segmenter_depth", type=int, default=12)

    parser.add_argument("--detr_token_dim", type=int, default=256)

    parser.add_argument("--detr_num_layers", type=int, default=4)

    parser.add_argument("--detr_num_heads", type=int, default=8)

    parser.add_argument("--detr_template_size", type=int, choices=[7, 13, 19], default=13)

    args = parser.parse_args()


    RANDOM_STATE = args.random_state

    random.seed(RANDOM_STATE)

    np.random.seed(RANDOM_STATE)

    torch.manual_seed(RANDOM_STATE)

    torch.cuda.manual_seed_all(RANDOM_STATE)


    SAMPLE_SIZES[:] = [

        int(x.strip()) for x in args.sample_sizes.split(",") if x.strip()

    ]

    NUM_EPOCHS = args.epochs

    if args.from_scratch:

        args.pretrained_ckpt = ""


    if torch.cuda.is_available():

        torch.cuda.reset_max_memory_allocated()

        torch.cuda.reset_accumulated_memory_stats()


    run_experiments(

        pre_trained=None,

        pretrained_ckpt=args.pretrained_ckpt,

        load_mode=args.load_mode,

        batch_size=args.batch_size,

        use_amp=not args.disable_amp,

        centroid_weight=args.centroid_weight,

        bce_weight=args.bce_weight,

        dice_weight=args.dice_weight,

        model_arch=args.model_arch,

        output_dir=args.output_dir,

        site_dia_label_dir=args.site_dia_label_dir,

        site_mask_weight=args.site_mask_weight,

        site_state_weight=args.site_state_weight,

        site_coord_weight=args.site_coord_weight,

        site_exist_weight=args.site_exist_weight,

        site_offset_reg_weight=args.site_offset_reg_weight,

        freeze_dwunet=args.freeze_dwunet,

        dia_num_ion_attn_layers=args.dia_num_ion_attn_layers,

        dia_token_dim=args.dia_token_dim,

        dia_num_heads=args.dia_num_heads,

        site_use_global_cross_attn=args.site_use_global_cross_attn,

        site_disable_tok_offset=args.site_disable_tok_offset,

        site_disable_psf_mask=args.site_disable_psf_mask,

        site_disable_coord_head=args.site_disable_coord_head,

        site_tok_offset_max_px=args.site_tok_offset_max_px,

        site_disable_recon_loss=args.site_disable_recon_loss,

        site_recon_weight_max=args.site_recon_weight_max,

        site_recon_warm_epochs=args.site_recon_warm_epochs,

        site_recon_ramp_epochs=args.site_recon_ramp_epochs,

        site_recon_no_warmup=args.site_recon_no_warmup,

        site_freeze_psf_sigma=args.site_freeze_psf_sigma,

        site_psf_kernel=args.site_psf_kernel,

        site_psf_beta_init=args.site_psf_beta_init,

        site_psf_profile=args.site_psf_profile,

        site_psf_sigma_init=args.site_psf_sigma_init,

        site_recon_lambda_ssim=args.site_recon_lambda_ssim,

        early_stopping_patience=args.early_stopping_patience,

        early_stopping_metric=args.early_stopping_metric,

        learning_rate=args.learning_rate,

        dense_calibration_mask=args.dense_calibration_mask,

        auto_pos_weight=args.auto_pos_weight,

        dense_base_channels=args.dense_base_channels,

        dense_vit_depth=args.dense_vit_depth,

        dense_patch_size=args.dense_patch_size,

        dense_setr_decoder=args.dense_setr_decoder,

        dense_segmenter_depth=args.dense_segmenter_depth,

        detr_token_dim=args.detr_token_dim,

        detr_num_layers=args.detr_num_layers,

        detr_num_heads=args.detr_num_heads,

        detr_template_size=args.detr_template_size,

    )
