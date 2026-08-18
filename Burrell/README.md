# Burrell spatial MLE baseline

This directory implements the method in A. H. Burrell et al., *Scalable
simultaneous multiqubit readout with 99.99% single-shot fidelity*, Phys. Rev.
A **81**, 040302 (2010), DOI: 10.1103/PhysRevA.81.040302.

## What is implemented

For each calibrated site, pixels are ranked by mean brightness and the first
`roi_size` pixels form the ROI. Calibration frames provide empirical discrete
count distributions (B_{νki}(n)) and (D_{νki}(n)), optionally conditioned on
the selected nearest-neighbour state configuration ν. A site is bright when

```text
sum_i log(B_{νki}(n_i) / D_{νki}(n_i)) >= 0.
```

The default prediction starts with every site dark and iteratively updates the
sites until the state vector is stable or `iterations` is reached. This is the
Burrell neighbour-conditioned iterative MLE idea; it is **not** crosstalk-matrix
inversion and uses no Gaussian PSF kernel, pseudoinverse, or Tikhonov term.

## Important calibration requirement

`fit_burrell` expects separate calibration frames with known bright and dark
states. Do not fit distributions or choose ROI size using the held-out test
frames. In a paper experiment, choose `roi_size`, neighbour count, histogram
smoothing (`alpha`), and any coordinate/ROI convention on validation data, then
freeze them before test evaluation. The current repository does not contain
the raw image arrays or a run manifest, so this package does not claim to
reproduce the manuscript's 0.8954 number by itself.

When mixed-state calibration shots and their known state vectors are available,
use `fit_burrell_mixed(images, states, coords, ...)`. It estimates separate
empirical distributions for each target-state/neighbor-state configuration,
with target-state back-off when a configuration is absent. This corresponds to
Burrell's (B_{\nu ki}) and (D_{\nu ki}) construction. The simpler
`fit_burrell(bright, dark, ...)` API is a pooled-data fallback and does not
contain information about mixed neighbour configurations.

### Why `n_neighbours=2`

The default is two nearest calibrated sites. This is the most defensible
literature-faithful setting: Burrell et al. modelled nearest-neighbour
state-dependent distributions and reported no measurable improvement when the
third neighbour was added. The choice is therefore a fixed baseline protocol,
not a claim that two neighbours are optimal for every two-dimensional register.
If the paper reports a 2-D-specific optimum, that requires a pre-registered
validation ablation over neighbour counts; it must not be selected on the test
set.

## Minimal use

```python
import numpy as np
from burrell_mle import fit_burrell, predict, accuracy

model = fit_burrell(
    bright_calibration, dark_calibration, site_coordinates,
    roi_size=10, n_neighbours=2, iterations=8,
)
prediction = predict(model, test_images)
print(accuracy(prediction, test_states))
```

The implementation uses empirical count histograms with additive smoothing
(`alpha=0.5`) only to avoid zero-probability logs. This is a numerical
stabilizer, not a physical crosstalk regularizer. For a strict reproduction,
set `alpha` and the ROI/neighbour protocol to the values recorded in the
experiment manifest.
