import numpy as np
from burrell_mle import fit_burrell_mixed, predict


def test_mixed_conditioned_readout_runs():
    rng = np.random.default_rng(7)
    states = rng.integers(0, 2, size=(120, 3), dtype=np.uint8)
    images = rng.poisson(2 + 12 * states[:, None, :].sum(axis=2)[:, :, None], size=(120, 5, 7))
    # The synthetic image is intentionally simple; this is an API/smoke test,
    # not an accuracy claim about the experimental platform.
    coords = np.array([[2, 2], [2, 4], [2, 6]], dtype=float)
    model = fit_burrell_mixed(images, states, coords, roi_size=3, n_neighbours=2)
    out = predict(model, images)
    assert out.shape == states.shape
    assert np.isin(out, [0, 1]).all()
