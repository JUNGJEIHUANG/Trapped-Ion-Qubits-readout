# Paper-aligned baseline experiments

## Dense segmentation: 96 training runs

`dense_protocol.py` is the source of truth for the three stages:

| Stage | Runs | Selection boundary |
|---|---:|---|
| Capacity/output resolution | 21 | seed 1, LR `1e-4`, BCE+Dice |
| LR/objective | 60 | 12 candidates for each Stage-1 winner |
| Final retraining | 15 | five frozen winners x seeds `{1,2,3}` |

Every candidate reselects evaluation radius `{2,3,4,5}` and threshold
`{0.30,...,0.70}` on validation. Threshold maximizes Youden's J within a
radius; candidate score is the best validation per-ion accuracy across radii.
The architecture, radius, and threshold are frozen before test use.

The in-repository models are U-Net (`standard_unet`), TransUNet-style
(`vit_unet`), SETR-style (`setr`), and Segmenter (`segmenter`). `train_main.py`
exposes every searched parameter and writes `candidate_result.json`.

## Official external implementations

SegMAN-T uses the official Tiny implementation with an ImageNet-1K encoder.
It reflection-pads `456x88` inputs to `480x96`, crops before decoding, and uses
no connected-component filtering. Supply an adapter through
`--segman-command`; it must write at least:

```json
{
  "val_ion_acc": 0.0,
  "decoder_radius": 3,
  "decoder_threshold": 0.49,
  "checkpoint": "outputs/segman_t/checkpoint.pth"
}
```

The DEIM row is a register-preserving D-FINE + MAL adaptation, not unmodified
DEIM. PSF-derived boxes are used; mosaic and mixup are disabled; Dense O2O is
absent; detections are assigned to the nearest calibrated site after inference.

- DEIM: <https://github.com/ShihuaHuang95/DEIM>
- SegMAN: <https://github.com/yunxiangfu2001/SegMAN>

Record upstream commit IDs, environments, and command lines with every run. Do
not relabel unmodified upstream results as the manuscript's adapted row.

## Detection searches

`detection_protocol.py` enumerates 216 DETR-style Query configurations, 216
D-FINE + MAL configurations, and the 12-configuration Site-AIT control grid.
`run_detection_protocol.py` performs a safe dry plan by default and supports
resumable execution. The in-repository DETR path exposes learning rate, token
dimension, depth, heads, and template size; D-FINE + MAL is supplied through an
explicit upstream adapter command.

## Dry-run verification

```bash
python -m experiments.run_dense_protocol --output-root outputs/protocol_check
python -m experiments.run_detection_protocol --output-root outputs/detection_check
python -m pytest experiments/tests -q
```
