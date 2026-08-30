# Site-AIT: physics-informed 300-ion fluorescence readout

This repository is the code-and-checkpoint release for Site-Adaptive Ion
Tokenization (Site-AIT). One latent token is bound to each calibrated ion site;
offset-corrected adaptive Top-16 sampling gathers local evidence, cross-site
self-attention models register-wide crosstalk, and a differentiable measured-PSF
objective constrains training without changing the inference graph.

## Release contents

- Site-AIT training/evaluation: `train_main.py`, `Testset_eval.py`,
  `Ablation_eval.py`, `Robustness_eval.py`, `train_retrained_ablation.py`.
- Model and physics modules: `nets/`, `loss.py`, `psf_kernels.py`,
  `perturbations.py`.
- Detector-aware self-supervised pretraining: `Pre_train/`.
- Released checkpoints: `Pretrain_weight/best.pth` and
  `Site_AIT_weight/seed_{1,2,3}/best.pth`.
- Current-paper baselines: matched filter and IPM in
  `Reproduce_baseline_code/`; DETR-style Query, U-Net,
  TransUNet-style, SETR-style, and Segmenter in `nets/`; and the
  dense-baseline search protocol in `experiments/`.

The benchmark additionally uses the authors' public D-FINE/DEIM and SegMAN
implementations. They are not vendored here; the adaptation boundary and
command interface are documented in `experiments/README.md`.

## Installation and verification

Use Python 3.10 or later and install the CUDA-compatible PyTorch build first:

```bash
pip install torch torchvision
pip install -r requirements.txt
python smoke_tests/run_all.py
python -m pytest experiments/tests Reproduce_baseline_code/tests -q
```

## Main model quick start

```bash
python train_main.py --model_arch site_dia \
  --pretrained_ckpt Pretrain_weight/best.pth --load_mode backbone \
  --site_dia_label_dir data/intersection_train_data/site_dia_labels \
  --sample_sizes 50000 --epochs 100 --batch_size 48 \
  --random_state 1 --output_dir outputs/site_ait_seed1
```

```bash
python Testset_eval.py --model_arch site_dia \
  --model_path Site_AIT_weight/seed_1/best.pth \
  --test_root data/intersection_test_data \
  --gt_all_bright_mask data/GroundTruth.png \
  --result_dir outputs/site_ait_eval_seed1
```

## Dense-baseline protocol

Complete 96-run Dense-baseline hyperparameters search protocol for fair competitions:

```bash
python -m experiments.run_dense_protocol --output-root outputs/dense_protocol
```

Execute it after supplying the training corpus, calibrated all-bright
mask, and a command template for the official SegMAN-T checkout:

```bash
python -m experiments.run_dense_protocol --execute \
  --data-root data/intersection_train_data \
  --calibration-mask data/GroundTruth.png \
  --segman-command "python external/segman_adapter.py --output {output_dir} --seed {seed} --lr {lr} --objective {objective} --stride {output_stride} --pretrained {imagenet_pretrained}"
```

The runner is resumable. Search sees only the fixed 40,000/10,000
train/validation split; `Real1600-Aug400` is not accepted by the search runner.

The detection search is separately enumerated and resumable:

```bash
python -m experiments.run_detection_protocol
```

Its dry plan contains 216 DETR-style Query candidates, 216 D-FINE + MAL
candidates, and the 12-candidate Site-AIT control grid. Execution requires an
explicit adapter command for the official DEIM checkout.

## License

See `LICENSE`.
