import numpy as np

from Reproduce_baseline_code.ipm import IonPixelMappingReadout


def test_ipm_fits_disjoint_calibration_ranges():
    rng = np.random.default_rng(4)
    centers = np.array([[4.0, 4.0], [11.0, 4.0]])
    dark = rng.normal(0.0, 0.01, size=(12, 9, 16))
    bright = dark.copy()
    bright[:, 4, 4] += 1.0
    bright[:, 4, 11] += 1.0
    model = IonPixelMappingReadout(centers).fit(bright, dark)
    assert model.predict(bright).all()
    assert not model.predict(dark).any()
    assert all(1 <= len(pixels) <= 90 for pixels in model.pixel_sets)
