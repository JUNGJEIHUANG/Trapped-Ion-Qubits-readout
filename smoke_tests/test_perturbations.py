"""Shape/range-invariant smoke test for the P1-P6 controlled-perturbation

operators (perturbations.py, App. I).

"""

import _pathsetup  # noqa: F401

import numpy as np

import perturbations as P


def _make_scene(seed=0):

    rng = np.random.default_rng(seed)

    H, W, K = 60, 80, 15

    image = rng.random((H, W)).astype(np.float32)

    coords = np.stack(

        [rng.uniform(8, W - 8, K), rng.uniform(8, H - 8, K)], axis=1

    ).astype(np.float32)

    valid = np.ones(K, dtype=np.float32)

    return rng, image, coords, valid


def test_p1_coordinate_drift():

    rng, _, coords, _ = _make_scene()

    for sigma in P.P1_LEVELS:

        out = P.p1_coordinate_drift(coords, sigma, rng=rng)

        assert out.shape == coords.shape

        if sigma == 0:

            assert np.allclose(out, coords)


def test_p2_psf_broadening():

    _, image, _, _ = _make_scene()

    for sigma in P.P2_LEVELS:

        out = P.p2_psf_broadening(image, sigma)

        assert out.shape == image.shape

        if sigma == 0:

            assert np.allclose(out, image)


def test_p3_snr_reduction():

    rng, image, _, _ = _make_scene()

    for rho in P.P3_LEVELS:

        out = P.p3_snr_reduction(image, rho, rng=rng)

        assert out.shape == image.shape

        assert out.min() >= 0.0 and out.max() <= 1.0

        if rho == 1.0:

            assert np.allclose(out, image)


def test_p4_hot_pixel_injection():

    rng, image, _, _ = _make_scene()

    for eta in P.P4_LEVELS:

        out = P.p4_hot_pixel_injection(image, eta, saturation_value=1.0, rng=rng)

        num_saturated = int((out >= 1.0).sum())

        expected = int(round(eta * image.size))

        assert abs(num_saturated - expected) <= 1

        if eta == 0:

            assert np.allclose(out, image)


def test_p5_single_site_rearrangement():

    rng, image, coords, valid = _make_scene()

    for delta_r in P.P5_LEVELS:

        out, meta = P.p5_single_site_rearrangement(image, coords, valid, delta_r, rng=rng)

        assert out.shape == image.shape

        if delta_r == 0:

            assert meta is None

            assert np.allclose(out, image)

        else:

            assert meta is not None and 0 <= meta["site_index"] < coords.shape[0]


def test_p6_anisotropic_psf_elongation():

    _, image, _, _ = _make_scene()

    for ratio in P.P6_LEVELS:

        out = P.p6_anisotropic_psf_elongation(image, ratio, sigma_x=1.0)

        assert out.shape == image.shape


if __name__ == "__main__":

    test_p1_coordinate_drift()

    test_p2_psf_broadening()

    test_p3_snr_reduction()

    test_p4_hot_pixel_injection()

    test_p5_single_site_rearrangement()

    test_p6_anisotropic_psf_elongation()

    print("PASS: test_perturbations")

