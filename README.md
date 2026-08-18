# Site-AIT: 300-Ion Fluorescence Readout

This repository contains the functional training, evaluation, and released
checkpoint code for Site-Adaptive Ion Tokenization (Site-AIT), a
physics-informed model for joint bright/dark-state inference from fluorescence
images of a calibrated 300-ion array. The implementation combines
offset-corrected Top-16 site tokenization, lattice-aware cross-site attention,
and a differentiable PSF reconstruction loss.

## Included

- Supervised training and evaluation: `train_main.py`, `Testset_eval.py`,
  `Ablation_eval.py`, `Robustness_eval.py`, and
  `train_retrained_ablation.py`.
- Model, data, loss, PSF, and perturbation modules: `nets/`, `dataset.py`,
  `loss.py`, `trainer.py`, `psf_kernels.py`, and `perturbations.py`.
- Self-supervised pretraining implementation: `Pre_train/`.
- Classical Burrell baseline: `Burrell/`.
- CPU-only smoke tests: `smoke_tests/run_all.py`.
- Released weights:
  - `Pretrain_weight/best.pth` (self-supervised pretraining checkpoint);
  - `Site_AIT_weight/seed_{1,2,3}/best.pth` (three Site-AIT checkpoints).

Datasets, raw annotations, generated outputs, and locally trained checkpoints
are deliberately not included. Their expected layout and all reproducible
commands are documented in `REPRODUCIBILITY.md`.

## Installation

Use Python 3.10 or later. Install the appropriate PyTorch build for your CUDA
environment first, then install the remaining dependencies:

```bash
pip install torch torchvision
pip install -r requirements.txt
```

Run the no-data smoke tests before a full experiment:

```bash
python smoke_tests/run_all.py
```

## Quick start

Set `IMG_DIR` and `SITE_DIA_LABEL_DIR` to the supervised dataset locations, then
train or evaluate from the repository root. For example:

```bash
python train_main.py --model_arch site_dia \
  --pretrained_ckpt Pretrain_weight/best.pth --load_mode backbone \
  --site_dia_label_dir /path/to/intersection_train_data/site_dia_labels \
  --sample_sizes 50000 --epochs 100 --batch_size 48 \
  --output_dir outputs/site_dia_main
```

For evaluation, use any released Site-AIT checkpoint, for example
`Site_AIT_weight/seed_1/best.pth`:

```bash
python Testset_eval.py --model_arch site_dia \
  --model_path Site_AIT_weight/seed_1/best.pth \
  --test_root /path/to/intersection_test_data \
  --gt_all_bright_mask /path/to/GroundTruth.png \
  --result_dir outputs/site_dia_eval
```



## License

See `LICENSE`.
