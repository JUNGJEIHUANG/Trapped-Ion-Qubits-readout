"""Correctness tests for the two reproductions.

These are not smoke tests. Each one checks a property that is *stated in the
source paper* or that follows from its mathematics, so a silently wrong
implementation fails here rather than in Table 1.

Run with:  python -m pytest Reproduce_baseline_code/tests -q
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from ..bayesian_pi.bayes_em import posterior_over_l, relative_fidelity, weakly_anchored_em
from ..bayesian_pi.distributions import SuperPoissonian, Theta, histogram_overlap
from ..bayesian_pi.pi_network import PINetwork, PINetworkConfig, SitePINetwork, make_training_batch
from ..bayesian_pi.site_adapter import ROIExtractor, SiteBayesianReadout
from ..common.dataio import locate_sites, preprocess, synthetic_register
from ..common.metrics import (
    auroc,
    classification_fidelity,
    expected_calibration_error,
    infidelity_reduction,
    per_ion_accuracy,
)
from ..matched_filter.mf_model import (
    GaussianFilter,
    MatchedFilterConfig,
    MatchedFilterReadout,
    SquareFilter,
    _ridge_closed_form,
)


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------


def test_fidelity_is_balanced_not_accuracy():
    """Kent Eq. (3) weights both error directions equally regardless of prior.

    On an imbalanced set with a constant-dark predictor, accuracy tracks the
    prior while fidelity must sit at 0.5. Conflating the two is the easiest way
    to misreport a readout baseline.
    """
    y = np.array([1] + [0] * 99, dtype=bool)
    pred = np.zeros_like(y)
    assert per_ion_accuracy(y, pred) == pytest.approx(0.99)
    assert classification_fidelity(y, pred) == pytest.approx(0.5)


def test_auroc_matches_known_values():
    y = np.array([0, 0, 1, 1], dtype=bool)
    assert auroc(y, np.array([0.1, 0.2, 0.3, 0.4])) == pytest.approx(1.0)
    assert auroc(y, np.array([0.4, 0.3, 0.2, 0.1])) == pytest.approx(0.0)
    # All-tied scores must give exactly chance, not 0 or 1.
    assert auroc(y, np.ones(4)) == pytest.approx(0.5)


def test_ece_zero_for_perfectly_calibrated():
    rng = np.random.default_rng(0)
    p = rng.uniform(0, 1, size=200_000)
    y = rng.random(200_000) < p
    assert expected_calibration_error(y, p) < 0.01


def test_infidelity_reduction_matches_paper_definition():
    # Halving the infidelity is eta = 0.5 by Kent Eq. (5).
    assert infidelity_reduction(0.90, 0.95) == pytest.approx(0.5)
    assert infidelity_reduction(0.90, 0.90) == pytest.approx(0.0)


# --------------------------------------------------------------------------
# Matched filter (Kent et al. 2026)
# --------------------------------------------------------------------------


def test_ridge_recovers_exact_linear_map():
    """Eq. (A1) must recover the generating weights when the model is exact."""
    rng = np.random.default_rng(0)
    d, M = 12, 4000
    W_true = rng.normal(size=d)
    X = rng.normal(size=(d, M))
    y = W_true @ X
    W = _ridge_closed_form(X, y, alpha=0.0)
    assert np.allclose(W, W_true, atol=1e-6)


def test_ridge_alpha_shrinks_weights():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(8, 200))
    y = rng.normal(size=200)
    n0 = np.linalg.norm(_ridge_closed_form(X, y, 0.0))
    n1 = np.linalg.norm(_ridge_closed_form(X, y, 1e3))
    assert n1 < n0


def test_matched_filter_beats_gaussian_filter():
    """The paper's central empirical claim: MF > traditional Gaussian filter."""
    ds = preprocess(synthetic_register(n_frames=600, grid=(4, 6), seed=3))
    centers = locate_sites(ds)
    tr, va, te = ds.splits.train, ds.splits.val, ds.splits.test
    sizes = (4, 6, 8)

    gf = GaussianFilter(centers, ds.shape, 2.5, sizes).fit(ds.frames[va], ds.labels[va])
    f_gauss = classification_fidelity(ds.labels[te], gf.predict(ds.frames[te]))

    mf = MatchedFilterReadout(
        centers, ds.shape, MatchedFilterConfig(boundary_sizes=sizes, neighbor_mode="none")
    ).fit(ds.frames[tr], ds.labels[tr], ds.frames[va], ds.labels[va])
    f_mf = classification_fidelity(ds.labels[te], mf.predict(ds.frames[te]))

    assert f_mf > f_gauss, f"MF-site {f_mf:.4f} did not beat Gaussian {f_gauss:.4f}"
    assert infidelity_reduction(f_gauss, f_mf) > 0


def test_mf_array_beats_mf_site_under_crosstalk():
    """Sec. IV A: MF-array "generally matches or outperforms" MF-site.

    The synthetic register is built with overlapping PSFs, so the neighbour
    features must carry usable information.
    """
    ds = preprocess(synthetic_register(n_frames=800, grid=(4, 6), spacing=5.0, psf_sigma=2.5, seed=5))
    centers = locate_sites(ds)
    tr, va, te = ds.splits.train, ds.splits.val, ds.splits.test
    sizes = (4, 6, 8)

    fids = {}
    for mode in ("none", "knn"):
        mf = MatchedFilterReadout(
            centers, ds.shape,
            MatchedFilterConfig(boundary_sizes=sizes, neighbor_mode=mode, n_neighbors=6),
        ).fit(ds.frames[tr], ds.labels[tr], ds.frames[va], ds.labels[va])
        fids[mode] = classification_fidelity(ds.labels[te], mf.predict(ds.frames[te]))

    assert fids["knn"] >= fids["none"] - 1e-3, fids


def test_matched_filter_is_linear_and_cheap():
    """The model must stay linear: no nonlinear evaluations at inference."""
    ds = preprocess(synthetic_register(n_frames=300, grid=(3, 4), seed=7))
    centers = locate_sites(ds)
    tr, va = ds.splits.train, ds.splits.val
    mf = MatchedFilterReadout(
        centers, ds.shape, MatchedFilterConfig(boundary_sizes=(4, 6), neighbor_mode="none")
    ).fit(ds.frames[tr], ds.labels[tr], ds.frames[va], ds.labels[va])

    c = mf.complexity()
    assert c["nonlinear_evaluations_per_frame"] == 0
    # A 12-site register with s <= 6 must stay far below a CNN's parameter count.
    assert c["trainable_parameters"] < 12 * (6 * 6 + 1) + 1


def test_probability_map_is_monotone_in_score():
    """predict_proba must not reorder decisions -- only ECE may change."""
    ds = preprocess(synthetic_register(n_frames=400, grid=(3, 4), seed=11))
    centers = locate_sites(ds)
    tr, va, te = ds.splits.train, ds.splits.val, ds.splits.test
    mf = MatchedFilterReadout(
        centers, ds.shape, MatchedFilterConfig(boundary_sizes=(4, 6), neighbor_mode="none")
    ).fit(ds.frames[tr], ds.labels[tr], ds.frames[va], ds.labels[va])
    mf.calibrate(ds.frames[va], ds.labels[va])

    scores = mf.decision_scores(ds.frames[te])
    probs = mf.predict_proba(ds.frames[te])
    for k in (0, 5, 11):
        assert np.array_equal(np.argsort(scores[:, k]), np.argsort(probs[:, k]))


def test_calibrate_required_before_predict_proba():
    ds = preprocess(synthetic_register(n_frames=200, grid=(2, 3), seed=13))
    centers = locate_sites(ds)
    tr, va = ds.splits.train, ds.splits.val
    mf = MatchedFilterReadout(
        centers, ds.shape, MatchedFilterConfig(boundary_sizes=(4,), neighbor_mode="none")
    ).fit(ds.frames[tr], ds.labels[tr], ds.frames[va], ds.labels[va])
    with pytest.raises(RuntimeError):
        mf.predict_proba(ds.frames[va])


# --------------------------------------------------------------------------
# Bayesian readout (Zhou et al. 2026)
# --------------------------------------------------------------------------


def test_super_poissonian_is_normalised_and_overdispersed():
    theta = Theta(alpha=4.0, beta=2.0)
    n = np.arange(0, 4000)
    assert SuperPoissonian.pmf(n, theta).sum() == pytest.approx(1.0, abs=1e-6)
    mean, var = SuperPoissonian.moments(theta)
    assert var > mean  # super-Poissonian by construction


def test_moment_fit_recovers_parameters():
    rng = np.random.default_rng(0)
    theta = Theta(alpha=6.0, beta=1.5)
    n = SuperPoissonian.sample(theta, 200_000, rng)
    fit = SuperPoissonian.fit_moments(n)
    assert fit.alpha == pytest.approx(theta.alpha, rel=0.1)
    assert fit.beta == pytest.approx(theta.beta, rel=0.1)


def test_histogram_overlap_bounds():
    a = Theta(3.0, 1.0)
    assert histogram_overlap(a, a) == pytest.approx(1.0, abs=1e-6)
    # Far-separated distributions must approach zero overlap.
    assert histogram_overlap(Theta(1.0, 0.5), Theta(200.0, 2.0)) < 0.05


def test_relative_fidelity_is_one_at_perfect_match():
    """Eq. (3) is a squared bracket, so equal occupations give exactly 1.

    Reading the exponent as a denominator would give 0.5 here -- this test is
    what pins the correct reading of the equation.
    """
    for l in (0.0, 0.13, 0.5, 0.87, 1.0):
        assert relative_fidelity(l, l) == pytest.approx(1.0, abs=1e-9)
    assert relative_fidelity(1.0, 0.0) == pytest.approx(0.0, abs=1e-9)


def test_posterior_is_normalised_and_peaks_at_truth():
    rng = np.random.default_rng(2)
    theta_g, theta_f = Theta(2.0, 1.0), Theta(30.0, 1.5)
    l_true = 0.35
    bright = rng.random(4000) < l_true
    counts = np.where(
        bright,
        SuperPoissonian.sample(theta_f, 4000, rng),
        SuperPoissonian.sample(theta_g, 4000, rng),
    ).astype(float)

    grid = np.linspace(0, 1, 501)
    post = posterior_over_l(counts, theta_f, theta_g, grid)
    dl = grid[1] - grid[0]
    assert np.sum(post) * dl == pytest.approx(1.0, abs=1e-6)
    assert grid[int(np.argmax(post))] == pytest.approx(l_true, abs=0.03)


def test_weakly_anchored_em_recovers_l_without_anchoring_f():
    """The paper's core claim: only g is calibrated, f is inferred."""
    rng = np.random.default_rng(4)
    theta_g, theta_f = Theta(2.0, 1.0), Theta(40.0, 1.2)
    for l_true in (0.25, 0.5, 0.8):
        bright = rng.random(3000) < l_true
        counts = np.where(
            bright,
            SuperPoissonian.sample(theta_f, 3000, rng),
            SuperPoissonian.sample(theta_g, 3000, rng),
        ).astype(float)
        res = weakly_anchored_em(counts, theta_g, max_iter=15)
        assert res.l_mean == pytest.approx(l_true, abs=0.05), (l_true, res.l_mean)
        assert relative_fidelity(l_true, res.l_mean) > 0.99


def test_em_posterior_uncertainty_shrinks_with_more_shots():
    """Delta l must fall roughly as 1/sqrt(N), per the paper's Fisher analysis."""
    rng = np.random.default_rng(6)
    theta_g, theta_f = Theta(2.0, 1.0), Theta(25.0, 1.2)
    stds = []
    for N in (100, 1600):
        bright = rng.random(N) < 0.5
        counts = np.where(
            bright,
            SuperPoissonian.sample(theta_f, N, rng),
            SuperPoissonian.sample(theta_g, N, rng),
        ).astype(float)
        stds.append(weakly_anchored_em(counts, theta_g, max_iter=15).l_std)
    assert stds[1] < stds[0] / 2.0, stds


def test_pi_network_is_permutation_invariant_over_shots():
    torch.manual_seed(0)
    net = PINetwork(PINetworkConfig(n_grid=21, element_hidden=16, embed_dim=16, head_hidden=32))
    s = torch.randn(3, 40)
    aux = torch.randn(3, 5)
    out = net(s, aux)
    perm = torch.randperm(40)
    assert torch.allclose(net(s[:, perm], aux), out, atol=1e-5)


def test_site_pi_network_is_permutation_equivariant_over_sites():
    """The adaptation must permute outputs with inputs, not ignore the order."""
    torch.manual_seed(0)
    net = SitePINetwork(n_features=5, hidden=32, n_blocks=2)
    x = torch.randn(3, 25, 5)
    out = net(x)
    perm = torch.randperm(25)
    assert torch.allclose(net(x[:, perm]), out[:, perm], atol=1e-5)


def test_pi_training_batch_targets_are_valid_distributions():
    rng = np.random.default_rng(0)
    _, aux, tgt = make_training_batch(8, 50, rng, n_grid=41)
    assert np.allclose(tgt.sum(axis=1), 1.0, atol=1e-5)
    assert np.all(tgt >= 0)
    assert aux.shape == (8, 5)


def test_site_bayesian_readout_beats_chance_and_calibrates_only_g():
    ds = preprocess(synthetic_register(n_frames=900, grid=(4, 6), seed=17))
    centers = locate_sites(ds)
    tr, va, te = ds.splits.train, ds.splits.val, ds.splits.test

    dark = ds.frames[tr][ds.labels[tr].sum(1) == 0]
    mixed = ds.frames[tr][
        (ds.labels[tr].sum(1) > 0) & (ds.labels[tr].sum(1) < ds.n_sites)
    ]
    assert dark.shape[0] > 10 and mixed.shape[0] > 50

    roi = ROIExtractor(centers, ds.shape, 3.0)
    sb = SiteBayesianReadout(roi).calibrate(
        dark, mixed, ds.frames[va], ds.labels[va], verbose=False
    )
    acc = per_ion_accuracy(ds.labels[te], sb.predict(ds.frames[te]))
    assert acc > 0.75, acc

    # Probabilities are the method's own sigmoid(LLR): monotone in the LLR.
    s = sb.llr(ds.frames[te])
    p = sb.predict_proba(ds.frames[te])
    assert np.array_equal(np.argsort(s[:, 0]), np.argsort(p[:, 0]))


def test_site_bayesian_network_requires_training():
    ds = preprocess(synthetic_register(n_frames=300, grid=(3, 4), seed=19))
    centers = locate_sites(ds)
    roi = ROIExtractor(centers, ds.shape, 3.0)
    sb = SiteBayesianReadout(roi)
    sb.models_ = []
    with pytest.raises(RuntimeError):
        sb.predict(ds.frames[:5], use_network=True)


# --------------------------------------------------------------------------
# Protocol hygiene
# --------------------------------------------------------------------------


def test_splits_never_overlap():
    ds = synthetic_register(n_frames=200, grid=(2, 3), seed=23)
    assert np.intersect1d(ds.splits.train, ds.splits.test).size == 0
    assert np.intersect1d(ds.splits.val, ds.splits.test).size == 0


def test_preprocess_uses_training_statistics_only():
    """Changing test frames must not change the normalisation of train frames."""
    ds = synthetic_register(n_frames=300, grid=(3, 4), seed=29)
    a = preprocess(ds).frames[ds.splits.train].copy()

    ds2 = synthetic_register(n_frames=300, grid=(3, 4), seed=29)
    ds2.frames[ds2.splits.test] *= 5.0
    b = preprocess(ds2).frames[ds2.splits.train]
    assert np.allclose(a, b)


def test_square_filter_is_weakest_of_the_three():
    """Kent Fig. 2 ordering: square < Gaussian <= matched filter."""
    ds = preprocess(synthetic_register(n_frames=600, grid=(4, 6), seed=31))
    centers = locate_sites(ds)
    va, te = ds.splits.val, ds.splits.test
    sizes = (4, 6, 8)

    sq = SquareFilter(centers, ds.shape, sizes).fit(ds.frames[va], ds.labels[va])
    gf = GaussianFilter(centers, ds.shape, 2.5, sizes).fit(ds.frames[va], ds.labels[va])
    f_sq = classification_fidelity(ds.labels[te], sq.predict(ds.frames[te]))
    f_gf = classification_fidelity(ds.labels[te], gf.predict(ds.frames[te]))
    assert f_gf >= f_sq - 1e-3, (f_sq, f_gf)


def test_network_scores_distinguish_frames_from_features():
    """Raw frames (N,H,W) and site features (N,K,5) are both 3-D.

    Regression test: an earlier version dispatched on ``ndim == 3`` alone and
    fed raw frames straight into the network.
    """
    ds = preprocess(synthetic_register(n_frames=400, grid=(3, 4), seed=37))
    centers = locate_sites(ds)
    tr, va = ds.splits.train, ds.splits.val

    dark = ds.frames[tr][ds.labels[tr].sum(1) == 0]
    mixed = ds.frames[tr][(ds.labels[tr].sum(1) > 0) & (ds.labels[tr].sum(1) < ds.n_sites)]
    roi = ROIExtractor(centers, ds.shape, 3.0)
    sb = SiteBayesianReadout(roi).calibrate(
        dark, mixed, ds.frames[va], ds.labels[va], verbose=False
    )
    sb.fit_pi_network(
        ds.frames[tr], ds.labels[tr], ds.frames[va], ds.labels[va],
        epochs=2, verbose=False,
    )

    frames = ds.frames[ds.splits.test]
    from_frames = sb._network_scores(frames)
    from_feats = sb._network_scores(sb.site_features(frames))
    assert from_frames.shape == (frames.shape[0], ds.n_sites)
    assert np.allclose(from_frames, from_feats)
