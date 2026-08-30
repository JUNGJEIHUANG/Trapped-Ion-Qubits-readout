"""End-to-end driver: fit both reproduced baselines and report Table-1 metrics.

    # verify the implementations on the synthetic register (no data needed)
    python -m Reproduce_baseline_code.run_baselines --synthetic --seeds 1 2 3

    # run on the real corpus
    python -m Reproduce_baseline_code.run_baselines --npz path/to/site_ait.npz --seeds 1 2 3

The output is a table in the same columns as Table 1 of the manuscript, plus
the source-paper metrics (classification fidelity, infidelity reduction vs the
Gaussian filter, mean histogram overlap) that let a reviewer check the
reproduction is faithful before checking that it is beaten.
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np

from .bayesian_pi.site_adapter import ROIExtractor, SiteBayesianReadout
from .common.dataio import ReadoutDataset, load_npz, locate_sites, preprocess, synthetic_register
from .common.metrics import (
    classification_fidelity,
    cross_fidelity,
    infidelity_reduction,
    summarize,
)
from .matched_filter.mf_model import (
    GaussianFilter,
    MatchedFilterConfig,
    MatchedFilterReadout,
    SquareFilter,
)


def _estimate_sigma(ds: ReadoutDataset, centers: np.ndarray) -> float:
    """Circular-Gaussian width from the mean training frame (Kent Eq. (2)).

    The fit window is half the nearest-neighbour spacing. On a dense register a
    wider window contains the neighbouring peaks, and the single-site Gaussian
    then fits the blended profile and reports a badly biased sigma -- which
    propagates straight into the Gaussian filter that serves as the reference
    for the infidelity-reduction metric.
    """
    import warnings

    from scipy.optimize import OptimizeWarning, curve_fit

    mean_frame = ds.frames[ds.splits.train].mean(axis=0)
    H, W = mean_frame.shape

    d = np.linalg.norm(centers[:, None, :] - centers[None, :, :], axis=-1)
    np.fill_diagonal(d, np.inf)
    half_spacing = float(np.median(d.min(axis=1))) / 2.0
    win = max(int(np.floor(half_spacing)), 2)

    sigmas = []
    for r0, c0 in centers[: min(32, centers.shape[0])]:
        r_lo, r_hi = int(max(0, r0 - win)), int(min(H, r0 + win + 1))
        c_lo, c_hi = int(max(0, c0 - win)), int(min(W, c0 + win + 1))
        patch = mean_frame[r_lo:r_hi, c_lo:c_hi]
        rr, cc = np.mgrid[r_lo:r_hi, c_lo:c_hi]

        def model(coords, amp, sigma, offset):
            r, c = coords
            return (amp * np.exp(-((r - r0) ** 2 + (c - c0) ** 2) / (2 * sigma**2)) + offset).ravel()

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", OptimizeWarning)
                popt, _ = curve_fit(
                    model, (rr, cc), patch.ravel(),
                    p0=(patch.max() - patch.min(), 2.0, patch.min()), maxfev=5000,
                )
            if 0.3 < abs(popt[1]) < 10:
                sigmas.append(abs(popt[1]))
        except (RuntimeError, ValueError):
            continue
    return float(np.median(sigmas)) if sigmas else 2.0


def _select_roi_radius(ds, centers, candidates, val_frames, val_labels) -> float:
    """Pick the ROI radius on validation only, mirroring app:mask_to_site_protocol."""
    best = (-np.inf, candidates[0])
    for r in candidates:
        roi = ROIExtractor(centers, ds.shape, r)
        counts = roi.counts(val_frames)
        fids = []
        for k in range(counts.shape[1]):
            s = counts[:, k]
            thr = np.quantile(s, np.linspace(0.05, 0.95, 19))
            fids.append(max(classification_fidelity(val_labels[:, k], s >= t) for t in thr))
        score = float(np.mean(fids))
        if score > best[0]:
            best = (score, r)
    return best[1]


def run_seed(
    ds: ReadoutDataset,
    seed: int,
    fast: bool = False,
    verbose: bool = True,
    pi_net: bool = False,
    pi_epochs: int = 30,
) -> dict:
    """Fit every model for one seed and return its metrics.

    Only the stochastic parts of the pipeline depend on ``seed``. The matched
    filter is a closed-form least-squares fit and the Bayesian EM is
    deterministic given its inputs, so their seed spread comes from the data
    ordering alone -- which is exactly the point the manuscript makes about
    "deterministic classical estimators have no seed spread". The seed loop is
    kept so that any stochastic component added later is covered.
    """
    rng = np.random.default_rng(seed)
    tr, va, te = ds.splits.train, ds.splits.val, ds.splits.test
    tr = rng.permutation(tr)

    train_f, train_y, _ = ds.subset(tr)
    val_f, val_y, _ = ds.subset(va)
    test_f, test_y, test_v = ds.subset(te)

    centers = locate_sites(ds)
    sigma = _estimate_sigma(ds, centers)
    if verbose:
        print(f"[seed {seed}] centers refined, sigma_psf ~= {sigma:.2f} px")

    boundary_sizes = (4, 6, 8) if fast else tuple(range(2, 15))
    thresholds = tuple(np.round(np.arange(0.05, 1.0, 0.05), 2)) if fast else None
    results = {}
    predictions = {}

    # --- Traditional references (needed for the paper's eta metric) ----------
    sq = SquareFilter(centers, ds.shape, boundary_sizes).fit(val_f, val_y)
    results["Square filter"] = summarize(test_y, sq.predict(test_f), valid=test_v)

    gf = GaussianFilter(centers, ds.shape, sigma, boundary_sizes).fit(val_f, val_y)
    gauss_pred = gf.predict(test_f)
    results["Gaussian filter"] = summarize(test_y, gauss_pred, valid=test_v)
    predictions["Gaussian filter"] = gauss_pred
    f_gauss = results["Gaussian filter"]["fidelity"]

    # --- Kent et al. 2026: MF-site and MF-array ------------------------------
    for name, mode in (("Matched filter (MF-site)", "none"), ("Matched filter (MF-array)", "knn")):
        cfg = MatchedFilterConfig(
            boundary_sizes=boundary_sizes,
            neighbor_mode=mode,
            n_neighbors=8,
            alphas=(0.0, 1e-6, 1e-4),
        )
        if thresholds is not None:
            cfg = MatchedFilterConfig(**{**cfg.__dict__, "thresholds": thresholds})

        t0 = time.time()
        mf = MatchedFilterReadout(centers, ds.shape, cfg)
        mf.fit(train_f, train_y, val_f, val_y, verbose=False)
        mf.calibrate(val_f, val_y)
        fit_s = time.time() - t0

        t0 = time.time()
        pred = mf.predict(test_f)
        infer_ms = 1000.0 * (time.time() - t0) / test_f.shape[0]

        m = summarize(test_y, pred, probs=mf.predict_proba(test_f), valid=test_v)
        m.update(
            eta_vs_gaussian=infidelity_reduction(f_gauss, m["fidelity"]),
            fit_seconds=fit_s,
            latency_ms_per_frame=infer_ms,
            **mf.complexity(),
        )
        results[name] = m
        predictions[name] = pred
        if verbose:
            print(f"[seed {seed}] {name}: acc={m['accuracy']:.4f} eta={m['eta_vs_gaussian']:+.3f}")

    # --- Zhou et al. 2026: weakly anchored Bayesian, site-adapted (Mode B) ---
    # g is anchored on all-dark frames only; f is inferred by EM on unlabelled
    # mixed frames. Frame-level all-dark selection uses the training labels'
    # frame sum, which is metadata the experiment records independently.
    all_dark = train_f[train_y.sum(axis=1) == 0]
    mixed = train_f[(train_y.sum(axis=1) > 0) & (train_y.sum(axis=1) < ds.n_sites)]
    if all_dark.shape[0] < 20 or mixed.shape[0] < 50:
        if verbose:
            print(f"[seed {seed}] skipping Bayesian readout: too few all-dark/mixed frames")
        return results

    radius = _select_roi_radius(ds, centers, (2.0, 3.0, 4.0, 5.0), val_f, val_y)
    roi = ROIExtractor(centers, ds.shape, radius)

    t0 = time.time()
    sb = SiteBayesianReadout(roi).calibrate(all_dark, mixed, val_f, val_y, verbose=False)
    fit_s = time.time() - t0

    t0 = time.time()
    pred = sb.predict(test_f)
    infer_ms = 1000.0 * (time.time() - t0) / test_f.shape[0]

    m = summarize(test_y, pred, probs=sb.predict_proba(test_f), valid=test_v)
    m.update(
        eta_vs_gaussian=infidelity_reduction(f_gauss, m["fidelity"]),
        roi_radius=radius,
        mean_histogram_overlap=sb.mean_overlap(),
        fit_seconds=fit_s,
        latency_ms_per_frame=infer_ms,
    )
    results["Bayesian LLR (site-adapted)"] = m
    predictions["Bayesian LLR (site-adapted)"] = pred
    if verbose:
        print(
            f"[seed {seed}] Bayesian LLR: acc={m['accuracy']:.4f} "
            f"radius={radius} overlap={m['mean_histogram_overlap']:.3f}"
        )

    # Register-coupled variant. The protocol requires both this and the
    # independent LLR above: the gap between them is the evidence for how much
    # of the task needs cross-site coupling.
    if pi_net:
        t0 = time.time()
        sb.fit_pi_network(
            train_f, train_y, val_f, val_y, epochs=pi_epochs, seed=seed, verbose=verbose
        )
        fit_s = time.time() - t0

        t0 = time.time()
        pred_net = sb.predict(test_f, use_network=True)
        infer_ms = 1000.0 * (time.time() - t0) / test_f.shape[0]

        m = summarize(
            test_y, pred_net,
            probs=sb.predict_proba(test_f, use_network=True), valid=test_v,
        )
        m.update(
            eta_vs_gaussian=infidelity_reduction(f_gauss, m["fidelity"]),
            roi_radius=radius,
            fit_seconds=fit_s,
            latency_ms_per_frame=infer_ms,
        )
        results["Bayesian LLR + site PI-Net"] = m
        predictions["Bayesian LLR + site PI-Net"] = pred_net
        if verbose:
            print(f"[seed {seed}] Bayesian LLR + PI-Net: acc={m['accuracy']:.4f}")

    # Crosstalk diagnostic (Kent Eq. (4)): predicted correlation between the
    # two closest sites, for every model. Computed on predictions only, so it
    # measures the model's own leakage rather than its accuracy.
    d = np.linalg.norm(centers[:, None] - centers[None, :], axis=-1)
    np.fill_diagonal(d, np.inf)
    k0 = int(np.argmin(d.min(axis=1)))
    k1 = int(np.argmin(d[k0]))
    results["_cross_fidelity_nearest_pair"] = {
        "sites": [k0, k1],
        **{name: cross_fidelity(p, k0, k1) for name, p in predictions.items()},
    }
    return results


def aggregate(per_seed: list) -> dict:
    """Mean +/- sample standard deviation across seeds, as Table 1 reports."""
    names = [k for k in per_seed[0] if not k.startswith("_")]
    out = {}
    for name in names:
        keys = [k for k in per_seed[0][name] if isinstance(per_seed[0][name][k], (int, float))]
        out[name] = {
            k: {
                "mean": float(np.mean([r[name][k] for r in per_seed])),
                "std": float(np.std([r[name][k] for r in per_seed], ddof=1)) if len(per_seed) > 1 else 0.0,
            }
            for k in keys
        }
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--npz", help="path to the real corpus")
    src.add_argument("--synthetic", action="store_true", help="run on the simulated register")
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--frames", type=int, default=3000, help="synthetic only")
    ap.add_argument("--grid", type=int, nargs=2, default=[10, 30], help="synthetic only")
    ap.add_argument("--fast", action="store_true", help="coarse metaparameter grids")
    ap.add_argument("--pi-net", action="store_true", help="also fit the site-equivariant PI-Network")
    ap.add_argument("--pi-epochs", type=int, default=30)
    ap.add_argument("--out", default=None, help="write aggregated results as JSON")
    args = ap.parse_args()

    if args.synthetic:
        print(f"simulating a {args.grid[0]}x{args.grid[1]} register, {args.frames} frames ...")
        ds = synthetic_register(n_frames=args.frames, grid=tuple(args.grid))
    else:
        ds = load_npz(args.npz)
    ds = preprocess(ds)
    print(f"frames {ds.frames.shape}, sites {ds.n_sites}, "
          f"splits {len(ds.splits.train)}/{len(ds.splits.val)}/{len(ds.splits.test)}")

    per_seed = [
        run_seed(ds, s, fast=args.fast, pi_net=args.pi_net, pi_epochs=args.pi_epochs)
        for s in args.seeds
    ]
    agg = aggregate(per_seed)

    print("\n" + "=" * 92)
    print(f"{'Model':<32}{'Acc.':>10}{'Prec.':>10}{'Recall':>10}{'F1':>10}{'Fidelity':>11}{'eta':>9}")
    print("-" * 92)
    for name, m in agg.items():
        eta = m.get("eta_vs_gaussian", {}).get("mean")
        print(
            f"{name:<32}{m['accuracy']['mean']:>10.4f}{m['precision']['mean']:>10.4f}"
            f"{m['recall']['mean']:>10.4f}{m['f1']['mean']:>10.4f}{m['fidelity']['mean']:>11.5f}"
            f"{('  n/a' if eta is None else f'{eta:>+9.3f}')}"
        )
    print("=" * 92)

    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(agg, fh, indent=2)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
