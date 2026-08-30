# Reproducibility Guide

This guide assumes the repository root is the current working directory.

## 1. Environment

Use Python 3.12 with a CUDA-enabled PyTorch installation when GPU training is available.

```bash
pip install torch torchvision
pip install -r requirements.txt
```

If your PyTorch build requires a specific CUDA wheel, install `torch` and `torchvision` from the official PyTorch selector first, then install the remaining dependencies from `requirements.txt`.

## 2. Expected data layout

By default, `config.py` looks for the training dataset at:

```text
data/intersection_train_data
```

The expected supervised training layout is:

```text
intersection_train_data/
  images/
    *.png
  masks/
    *.png
  site_dia_labels/
    site_coords.npy
    state_hard.npy
    labels_all.npz
    per_sample_npz/
      *.npz
```

You can override the dataset location with environment variables:

```bash
export IMG_DIR=data/intersection_train_data
export SITE_DIA_LABEL_DIR=data/intersection_train_data/site_dia_labels
```

On PowerShell:

```powershell
$env:IMG_DIR="data\intersection_train_data"
$env:SITE_DIA_LABEL_DIR="data\intersection_train_data\site_dia_labels"
```

## 3. Optional self-supervised pretraining

To create a pretraining checkpoint, run:

```bash
python Pre_train/main_train.py --data_dir data/unlabeled/images --save_dir Pre_train/Run_pretrain
```

After pretraining, pass the checkpoint path to supervised training with `--pretrained_ckpt`.

## 4. Main Site-AIT training

Train Site-AIT (mask-free, offset-corrected adaptive Top-16 tokenization + PSF reconstruction consistency, paper Sec. IV):

```bash
python train_main.py \
  --model_arch site_dia \
  --pretrained_ckpt Pre_train/Run_pretrain/best.pth \
  --load_mode backbone \
  --site_dia_label_dir data/intersection_train_data/site_dia_labels \
  --sample_sizes 50000 \
  --epochs 100 \
  --batch_size 48 \
  --output_dir outputs/site_dia_main
```

To train without a pretrained checkpoint:

```bash
python train_main.py \
  --model_arch site_dia \
  --from_scratch \
  --site_dia_label_dir data/intersection_train_data/site_dia_labels \
  --sample_sizes 50000 \
  --epochs 100 \
  --batch_size 48 \
  --output_dir outputs/site_dia_from_scratch
```

`--site_mask_weight` defaults to `0` because Site-AIT's inference path never predicts a dense mask (paper Sec. III); the flag exists only for the (unused by the paper) diagnostic mask head.

### Tokenization and reconstruction-loss flags (Sec. IV.C, IV.G, App. H)

| Flag | Default | Meaning |
| --- | --- | --- |
| `--dia_token_dim` | 256 | Lattice-aware site-token dimension `D` (paper: `D=256`). |
| `--dia_num_heads` | 8 | Attention heads for tokenization / ion-interaction (paper Sec. VI A, given). |
| `--site_tok_offset_max_px` | 3.0 | `R_Delta`, max per-axis tokenization offset in pixels (paper Sec. VI A, given). |
| `--site_psf_sigma_init` | 1.5 | Initial `sigma_psf` (px) for the PSF forward model (paper App. H, given). |
| `--site_recon_weight_max` | 0.05 | `lambda_recon^max` (paper App. H, given). |
| `--site_recon_warm_epochs` | 10 | `E_warm` (paper App. H, given). |
| `--site_recon_ramp_epochs` | 5 | `E_ramp` (paper App. H, given). |
| `--site_recon_lambda_ssim` | 0.2 | SSIM weight inside `L_recon` (paper Sec. VI A, given). |

### Table VI ablation flags

| Flag | Table VI row |
| --- | --- |
| `--site_use_global_cross_attn` | w/ global cross-attn |
| `--dia_num_ion_attn_layers 0` | w/o ion self-attn |
| `--site_disable_tok_offset` | w/o tokenization offsets |
| `--site_disable_psf_mask` | w/o learned PSF mask |
| `--from_scratch` | w/o phys. pretraining |
| `--site_disable_coord_head` | w/o learn. coord. constraint |
| `--site_disable_recon_loss` | Full - L_recon, no recon. loss |
| `--site_freeze_psf_sigma` | Full + L_recon, sigma_psf frozen |
| `--site_recon_no_warmup` | Full + L_recon, no warm-up |

Use `train_retrained_ablation.py` (Section 6) to train all 10 rows with one command.

## 5. Evaluation

Evaluate a trained checkpoint on a test set:

```bash
python Testset_eval.py \
  --model_arch site_dia \
  --model_path outputs/site_dia_main/sample_50000/best.pth \
  --test_root data/intersection_test_data \
  --gt_all_bright_mask data/GroundTruth.png \
  --result_dir outputs/site_dia_eval
```

`--model_arch detr_query` evaluates the current DETR-style Query baseline. `set_transformer` remains available only for legacy experiments and is not a current-table row (Section 8).

## 6. Ablation training and evaluation (Table VI)

Table VI has 10 independently-trained rows: `full`, an architectural block
(`global_cross_attn`, `no_self_attn`, `no_tok_offsets`, `no_psf_mask`, `no_pretrain`,
`no_coord_constraint`), and a PSF-reconstruction-consistency block
(`no_recon_loss`, `recon_sigma_frozen`, `recon_no_warmup`).

Train every variant as a separate checkpoint:

```bash
python train_retrained_ablation.py \
  --pretrained_ckpt Pre_train/Run_pretrain/best.pth \
  --site_dia_label_dir data/intersection_train_data/site_dia_labels \
  --sample_size 50000 \
  --epochs 100 \
  --batch_size 48 \
  --output_root outputs/retrained_ablation
```

Add `--only full,no_self_attn` to train a subset, or `--dry_run` to preview the commands without running them.

Evaluate all trained variants together:

```bash
python Ablation_eval.py \
  --full_ckpt outputs/retrained_ablation/full/sample_50000/best.pth \
  --global_cross_attn_ckpt outputs/retrained_ablation/global_cross_attn/sample_50000/best.pth \
  --no_self_attn_ckpt outputs/retrained_ablation/no_self_attn/sample_50000/best.pth \
  --no_tok_offsets_ckpt outputs/retrained_ablation/no_tok_offsets/sample_50000/best.pth \
  --no_psf_mask_ckpt outputs/retrained_ablation/no_psf_mask/sample_50000/best.pth \
  --no_pretrain_ckpt outputs/retrained_ablation/no_pretrain/sample_50000/best.pth \
  --no_coord_constraint_ckpt outputs/retrained_ablation/no_coord_constraint/sample_50000/best.pth \
  --no_recon_loss_ckpt outputs/retrained_ablation/no_recon_loss/sample_50000/best.pth \
  --recon_sigma_frozen_ckpt outputs/retrained_ablation/recon_sigma_frozen/sample_50000/best.pth \
  --recon_no_warmup_ckpt outputs/retrained_ablation/recon_no_warmup/sample_50000/best.pth \
  --test_root data/intersection_test_data \
  --gt_all_bright_mask data/GroundTruth.png \
  --result_dir outputs/ablation_eval
```

Pass `--only` with a comma-separated subset of variant keys to evaluate only the checkpoints you already have.

## 7. Basic validation

Before running long jobs, check syntax:

```bash
python -m py_compile train_main.py Testset_eval.py Ablation_eval.py Robustness_eval.py dataset.py loss.py trainer.py perturbations.py
python -m py_compile nets/*.py smoke_tests/*.py
```

Then run the shape/gradient-flow smoke tests (no GPU or dataset required):

```bash
python smoke_tests/run_all.py
```

These smoke tests verify the Site-AIT tensor paths, reconstruction loss, selected in-repository baselines, and perturbation operators on random tensors. The separate experiment tests verify dense-protocol accounting and validation decoding. They do **not** verify that training reproduces the paper's reported numbers — that requires the real dataset and a full training run.

## 8. Current-paper baselines (App. F)

The released benchmark groups comparators as classical estimators,
query-based detection baselines, and dense-segmentation baselines. The old Set
Transformer experiment remains in `nets/SetTransformer.py` only as a legacy
ablation; it is not a row in the current manuscript and must not be substituted
for any current result.

### 8.1 Matched filter

The MF-array implementation is in
`Reproduce_baseline_code/matched_filter/mf_model.py`. It uses training-only
centering/range normalization, Gaussian center refinement seeded by the nominal
lattice, eight nearest calibrated neighbors fixed a priori, per-site validation
selection of boundary width/regularization/threshold, and validation-fitted
positive-temperature probability calibration.

```bash
python -m Reproduce_baseline_code.run_baselines \
  --npz data/site_ait.npz --seeds 1 2 3 --out outputs/matched_filter.json
```

The package also contains a Bayesian readout research reproduction retained
from an earlier comparison. It is not a current Table-I row; do not report it as
one.

The current IPM implementation is in
`Reproduce_baseline_code/ipm/ipm_model.py`. It initializes each pixel set at a
local peak of the training bright--dark contrast image, grows pixels using the
frozen manuscript settings `xi=0.5`, `T_P=90`, and `T_K=4`, and freezes per-site
thresholds from calibration data before test evaluation.

### 8.2 DETR-style Query

`DETRStyleQuery` (`nets/DETRQuery.py`) shares the DW-UNet initialization and
nominal lattice with Site-AIT and trains with bright-state classification loss:

```bash
python train_main.py --model_arch detr_query \
  --pretrained_ckpt Pretrain_weight/best.pth --load_mode backbone \
  --site_dia_label_dir data/intersection_train_data/site_dia_labels \
  --sample_sizes 50000 --epochs 100 --batch_size 48 --random_state 1 \
  --output_dir outputs/detr_query_seed1
```

The exhaustive detection search spaces stated in the supplement are 216
configurations for DETR-style Query and 216 for the D-FINE + MAL adaptation;
the Site-AIT control grid contains 12 configurations. They are enumerated by
`experiments/detection_protocol.py` and run resumably by
`experiments/run_detection_protocol.py`. A dry run writes all 444 search
records without launching training:

```bash
python -m experiments.run_detection_protocol
```

Execution requires the 50,000-frame development corpus, site-level labels, the
released pretrained backbone, and an explicit adapter command for the official
DEIM checkout. These searches are distinct from the 96-run dense protocol.

### 8.3 Dense segmentation: executable 21+60+15 protocol

`experiments/run_dense_protocol.py` implements the complete dependent search:

1. 21 capacity/output-resolution runs at seed 1;
2. 60 learning-rate/objective runs (12 for each Stage-1 winner);
3. 15 frozen-winner retraining runs (five models x seeds 1, 2, and 3).

Each candidate performs nested validation-only disk-radius and threshold
selection through `experiments/dense_decoder.py`. Threshold is selected by
Youden's index for every radius in `{2,3,4,5}` and the candidate is scored by
its best validation per-ion accuracy. `Real1600-Aug400` is never passed to the
search runner.

Dry plan (writes exactly 96 JSONL records and launches no training):

```bash
python -m experiments.run_dense_protocol --output-root outputs/dense_protocol
```

Execution requires the fixed 50,000-frame development corpus (internally split
40,000/10,000), an all-bright calibration mask, and an adapter command for the
official SegMAN checkout:

```bash
python -m experiments.run_dense_protocol --execute \
  --data-root data/intersection_train_data \
  --calibration-mask data/GroundTruth.png \
  --segman-command "python external/segman_adapter.py --output {output_dir} --seed {seed} --lr {lr} --objective {objective} --stride {output_stride} --pretrained {imagenet_pretrained}"
```

`train_main.py` exposes U-Net base channels, TransUNet-style base channels and
ViT depth, SETR patch size/decoder (`pup`, `naive`, `mla`), Segmenter patch size
and encoder depth, learning rate, training-only positive-class weighting, and
validation-accuracy early stopping. It writes `candidate_result.json` for
resumable winner selection.

### 8.4 External DEIM and SegMAN boundaries

The manuscript's DEIM label denotes a register-preserving D-FINE + MAL
adaptation. Bright ions become PSF-derived boxes; mosaic and mixup are disabled;
Dense O2O is absent; detections are assigned to their nearest calibrated sites
after inference. It is not an unmodified DEIM run.

SegMAN-T uses the official Tiny implementation with an ImageNet-1K-pretrained
encoder. Frames are reflection-padded from `456x88` to `480x96`, outputs are
cropped before mask-to-site decoding, and no connected-component filtering is
used. The protocol runner intentionally requires an explicit external adapter
instead of silently shipping a divergent reimplementation. Pin and record the
upstream commit for every run:

- DEIM: <https://github.com/ShihuaHuang95/DEIM>
- SegMAN: <https://github.com/yunxiangfu2001/SegMAN>

## 9. Physical robustness evaluation (Sec. V.C, App. I, Table III / Fig. 3)

`perturbations.py` implements the six controlled-perturbation operators P1-P6 with the exact parameter grids from App. I. `Robustness_eval.py` applies each operator at each level to a trained checkpoint (zero-shot, no retraining, matching the paper's protocol) and writes a long-format CSV:

```bash
python Robustness_eval.py \
  --model_arch site_dia \
  --model_path outputs/site_dia_main/sample_50000/best.pth \
  --test_root data/intersection_test_data \
  --gt_all_bright_mask data/GroundTruth.png \
  --result_dir outputs/robustness_eval \
  --seeds 1,2,3
```

The manuscript-specific robustness plotting utility is maintained separately in
the paper workspace. This repository retains `Robustness_eval.py`, which writes
the measured long-format CSV required for that utility.

P5 (single-site rearrangement) is the one operator App. I describes only at the physical level; `perturbations.py::p5_single_site_rearrangement`'s pixel-level implementation (cut-and-repaste a local patch, refill the vacated region with a local background estimate) is documented as an engineering approximation, not an unambiguous reproduction of the paper's intent.


## Hyperparameters now stated by the paper

The paper is treated as the ground truth for every algorithmic detail. The following values were previously engineering defaults chosen here; the paper now states each of them explicitly, and the defaults below match it:

- `R_Delta` (tokenization offset cap, `--site_tok_offset_max_px`, default 3.0 px) — paper Sec. VI A, Implementation details.
- `sigma_psf` initialization (`--site_psf_sigma_init`, default 1.5 px) — paper App. H, PSF-consistency implementation.
- Attention head count for tokenization / ion-interaction (`--dia_num_heads`, default 8) — paper Sec. VI A, Implementation details ("a single transformer encoder layer with eight attention heads"); the same flag also sets the head count of the cross-attention baselines, which the paper does not specify separately.
- `lambda_ssim` inside `L_recon` (`--site_recon_lambda_ssim`, default 0.2) — paper Sec. VI A, Implementation details.

## Remaining assumptions not given explicit values by the paper

- The exact mechanism of the "w/ global cross-attn" ablation row (`nets/site_tokenizer.py::GlobalCrossAttnTokenizer`): implemented as full attention over the entire flattened feature map, consistent with the paper's reported higher latency for this row.
- The pixel-level mechanism of P5 (single-site rearrangement, `perturbations.py::p5_single_site_rearrangement`).

None of these settings can be verified against the paper's tables without the real dataset and a full training run, which this repository does not include.
