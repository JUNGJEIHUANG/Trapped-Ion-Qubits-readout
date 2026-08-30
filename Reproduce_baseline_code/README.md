# `Reproduce_baseline_code` — classical-baseline reproduction and retained research comparators


> **Current-manuscript scope.** MF-array and `ipm/` correspond to current Table-I rows. The Bayesian/PI-Network code is retained as an earlier research comparator and must not be reported as a current-paper baseline. Simulator values below validate code behavior only.
Two of the four new baselines ship no source code, so they are reimplemented
here from the papers:

| Reproduced | Source | Status |
|---|---|---|
| Matched filter (MF-site, MF-array) | Kent et al., *Phys. Rev. Applied* **25**, 044048 (2026) | complete; method fully specified in the paper |
| Weakly anchored Bayesian readout + PI-Network | Zhou et al., *Phys. Rev. Lett.* **137**, 013601 (2026) | complete, with **two documented assumptions** — see below |

The other two (DEIM, CVPR 2025; SegMAN, CVPR 2025) have official repositories
and are **not** reimplemented here. Use upstream code:
`https://github.com/ShihuaHuang95/DEIM` and
`https://github.com/yunxiangfu2001/SegMAN`.

The released experimental protocol and current-paper scope are documented in [`../REPRODUCIBILITY.md`](../REPRODUCIBILITY.md).

---

## Layout

```
Reproduce_baseline_code/
  common/
    dataio.py        dataset interface, preprocessing, site localisation, simulator
    metrics.py       Site-AIT metrics + the source papers' own metrics
  matched_filter/
    mf_model.py      MF-site, MF-array, and the square/Gaussian references
  bayesian_pi/
    distributions.py super-Poissonian photon-count model
    bayes_em.py      weakly anchored Bayesian-EM (the paper's own task)
    pi_network.py    PINetwork (invariant over shots), SitePINetwork (equivariant over sites)
    site_adapter.py  ROI extraction; Mode A (ensemble) and Mode B (single-frame)
    train_pi.py      KL distillation of the exact posterior
  run_baselines.py   end-to-end driver, prints a Table-1-shaped summary
  tests/             26 property tests tied to claims in the source papers
  conftest.py        OpenMP workaround for this machine (see Troubleshooting)
```

## Install and verify

```bash
pip install -r Reproduce_baseline_code/requirements.txt
```

```bash
python -m pytest Reproduce_baseline_code/tests -q
```

Expected: `26 passed`. The tests are property tests, not smoke tests — each
checks something the source paper states (e.g. MF-array ≥ MF-site under
crosstalk; the EM recovers `l` without anchoring `f`; the PI-Network is
permutation-invariant over shots and the site variant is equivariant over
sites), so a silently wrong implementation fails here rather than in Table 1.

## Run

On the built-in simulator, no data required:

```bash
python -m Reproduce_baseline_code.run_baselines --synthetic --frames 3000 --grid 10 30 --seeds 1 2 3
```

On the real corpus:

```bash
python -m Reproduce_baseline_code.run_baselines --npz path/to/site_ait.npz --seeds 1 2 3 --out results.json
```

The `.npz` must contain `frames (N,H,W)`, `labels (N,K) bool`,
`lattice (K,2)`, `train_idx`, `val_idx`, `test_idx`, and optionally
`valid (N,K)`. Nothing else in the package touches the file format.

Train the PI-Network (needed only for the ensemble-mode speedup claim):

```bash
python -m Reproduce_baseline_code.bayesian_pi.train_pi --steps 8000 --batch-size 128 --out pi_net.pt
```

---

## What was faithful, and what had to be decided

Everything traceable to a paper carries its section number in the source. Every
open choice is marked `ADAPTATION` in the code. The ones that matter:

### Matched filter — no substantive gaps

The paper specifies the method completely (Sec. III, Appendix A). Two small
decisions:

1. **Regularisation.** The paper sets `alpha = 0` for a 9-qubit, 28×28 problem.
   A 300-site register with `s` up to 14 gives a 197-dimensional feature vector
   per site and `X Xᵀ` becomes ill-conditioned far more often, so `alpha` is
   selected on validation from `{0, 1e-6, 1e-4}`. Setting `alphas=(0.0,)`
   reproduces the paper exactly.
2. **Neighbour set.** MF-array in the paper averages over *all* other qubits,
   which is fine for 9 and quadratic for 300. The paper itself sanctions the
   fix — "limiting the feature vector to include only nearest neighbors also
   results in linear scaling" (Sec. IV C) — so `neighbor_mode="knn"` with `k`
   selected on validation is the default. `neighbor_mode="all"` is available
   and is what should be used if you want the literal 3×3-array formulation.

`predict_proba` is an addition, not a change: the paper thresholds a raw linear
score and never needs a probability, but Table 1 reports AUROC and ECE. The map
is a per-site logistic recentred on the selected threshold, with its temperature
fitted on validation. It is monotone, so accuracy, precision, recall, F1 and
AUROC are untouched; only ECE depends on it. `predict_proba` raises unless
`calibrate()` has been called, so an uncalibrated model can never be reported
as calibrated by accident.

### Bayesian readout — two assumptions you must know about

**Assumption 1 — the super-Poissonian family.** The paper models `f` and `g` as
super-Poissonian with `theta = {alpha, beta}` but defers the functional form to
Supplemental Material, which is not in the article PDF. We use the
Gamma–Poisson mixture (negative binomial): two parameters, mean `alpha*beta`,
variance `alpha*beta*(1+beta)`, reduces to Poisson as `beta → 0`, and admits
the closed-form weighted moment estimation the M-step needs. This is the
standard EMCCD photon-count model. **If the SM specifies something else, only
`distributions.py` changes** — `bayes_em.py` and `pi_network.py` consume it
through `pmf` / `fit_moments` / `moments` / `sample` and are unaffected.

**Assumption 2 — Eq. (3) is a squared bracket.** Plain-text extraction of the
PDF renders the exponent on its own line, where it reads like a denominator.
The typeset equation on p. 3 was checked visually: it is
`F = [√(l_th·l̄) + √((1−l_th)(1−l̄))]²`. The denominator reading is also ruled
out physically — it would give `F = 0.5` for a perfect match and could never
produce the paper's reported `F ≥ 99.99%`. `test_relative_fidelity_is_one_at_perfect_match`
pins this.

### Bayesian readout — the task mismatch, and how it is handled

This is the most important thing in this package, and the protocol document
treats it at length.

Zhou et al. estimate `l`, the bright-state **occupation probability** of a
qubit, from **N repeated shots** of ROI-integrated photon counts. The output is
a continuous posterior with a calibrated uncertainty `Δl`. Site-AIT classifies
each of 300 sites as bright or dark from a **single frame**.

These are different estimands. Running the paper's estimator at `N = 1` and
thresholding it would guarantee a bad number and would be a strawman. So both
modes are implemented:

* **Mode A — `EnsembleReadout`.** The paper's own task, unchanged. Use it to
  validate the reproduction against the published numbers and to compare on the
  paper's own ground. Belongs in the supplementary material.
* **Mode B — `SiteBayesianReadout`.** The strongest faithful reading under
  Site-AIT's protocol. What carries over exactly: the *weakly anchored*
  calibration (only `g` is anchored, from all-dark frames; `f` is inferred by EM
  on unlabelled mixed frames, never from site labels), and the log-likelihood
  ratio `s_k = log[f(n_k)/g(n_k)]`, which is what the posterior reduces to for a
  single observation. Belongs in Table 1.

Mode B has two variants and **both should be reported**:

| Variant | Mechanism | Why report it |
|---|---|---|
| independent LLR | per-site decision on a scalar ROI sum | the literal single-frame reduction of the method |
| `SitePINetwork` | Deep Sets encoder, permutation-**equivariant** over sites, with a pooled register context | without it the method cannot represent crosstalk *at all*, and the comparison against Site-AIT's cross-site attention degenerates into "has global context vs. does not" |

`SitePINetwork` is the one genuine re-derivation in this package. The source
network is invariant over *shots* and returns one posterior over a scalar.
Site-AIT needs 300 outputs from one frame, so the required symmetry is
equivariance over *sites*: relabelling sites must permute the outputs. It is
built from the same Deep Sets ingredients the paper cites, and it consumes only
the LLR and the calibrated `(α_g, β_g, α_f, β_f)` per site — nothing from the
image that the source method does not already use.

---

## Observed behaviour on the simulator

An 8×12 register, 1200 frames, spacing/PSF = 2.4, 2 seeds, coarse grids:

```bash
python -m Reproduce_baseline_code.run_baselines --synthetic --frames 1200 --grid 8 12 \
    --seeds 1 2 --fast --pi-net --pi-epochs 15
```

| Model | Acc. | F1 | Fidelity | η vs Gaussian |
|---|---|---|---|---|
| Square filter | 0.9230 | 0.9247 | 0.92297 | — |
| Gaussian filter | 0.9262 | 0.9281 | 0.92605 | — |
| Matched filter (MF-site) | 0.9659 | 0.9666 | 0.96588 | +0.539 |
| Matched filter (MF-array) | **0.9681** | **0.9689** | **0.96809** | **+0.568** |
| Bayesian LLR (site-adapted) | 0.9243 | 0.9258 | 0.92431 | −0.024 |
| Bayesian LLR + site PI-Net | 0.9234 | 0.9259 | 0.92306 | −0.041 |

Three things to read from this:

1. **The matched-filter ordering reproduces**: square < Gaussian < MF-site <
   MF-array, matching Kent Fig. 2. The absolute η is larger here (+0.54/+0.57
   vs the paper's +0.32/+0.43) because the simulator is a different, harder
   register — the *ordering* is the reproduction check, not the magnitude.
2. **The independent Bayesian LLR lands at roughly the Gaussian filter.** Real
   result, not a bug: for a single frame the LLR is a monotone function of the
   scalar ROI sum, so it reduces to a photon-sum filter with a learned
   threshold and discards spatial structure entirely. This is the strongest
   argument for reporting the `SitePINetwork` variant alongside it rather than
   presenting the independent LLR on its own.
3. **The PI-Net variant gives no gain on *this* register.** Expected: with the
   simulator's mild leakage there is little for a register-level context to
   recover, and the network only sees scalar per-site LLRs. Whether it helps at
   the real spacing/PSF = 2.4 with 300 sites is an open question the real run
   answers — do not assume either way.

These are simulator numbers. They exist to show the code is correct and the
ordering is sane; they are **not** predictions for Table 1.

> **A note on σ estimation.** The Gaussian filter's σ is fitted on the mean
> training frame, and the fit window is capped at half the nearest-neighbour
> spacing — a wider window contains the neighbouring peaks and the fitted σ
> comes out badly biased, which propagates straight into the η reference. Even
> with the cap, a dense register truncates the profile: the run above recovers
> σ ≈ 1.56 px where the simulator's truth is 2.5. That is inherent to
> spacing/PSF = 2.4 and is exactly why an isolated-Gaussian filter is the wrong
> tool here. If you have an independent PSF calibration for the real register,
> pass it in rather than refitting.

## PI-Network distillation

The network is distilled against the exact posterior, so the honest success
criterion is agreement with that posterior, reported as mean KL and mean
absolute error in `l̄` on parameters never seen in training:

| Steps | Batch | Wall clock (CPU) | Held-out KL | Held-out `MAE(l̄)` |
|---|---|---|---|---|
| — (init) | — | — | 2.40 | 0.236 |
| 1,500 | 64 | 22 s | 0.323 | 0.0166 |
| 8,000 | 128 | 211 s | 0.118 | 0.0086 |

Still improving at 8,000 steps, so treat these as a floor, not a converged
result. The paper's claim is agreement "to within 0.1%", which needs longer
training **and** parameter sampling ranges narrowed to the actual calibration
rather than the broad defaults in `make_training_batch`. Narrow `alpha_range` /
`beta_range` to the fitted `(α_g, β_g, α_f, β_f)` of your register before
reading anything into the residual gap — the defaults deliberately span a much
wider regime than any single experiment occupies.

## Troubleshooting

**`OMP: Error #15` / `Fatal Python error: Aborted`.** Anaconda's MKL-linked
NumPy and pip-installed PyTorch each ship an Intel OpenMP runtime. `conftest.py`
sets `KMP_DUPLICATE_LIB_OK=TRUE` for the test suite; for scripts, export it
before running, or install both from one wheel family. This is an environment
property, not a code defect.

**`skipping Bayesian readout: too few all-dark/mixed frames`.** Mode B anchors
`g` on all-dark frames. The manuscript's training partition has 10,000 of them,
so this only appears on small synthetic runs.
