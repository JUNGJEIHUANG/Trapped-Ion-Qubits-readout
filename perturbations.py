"""Controlled-perturbation operators P1-P6 (Site-AIT paper App. I).

Each operator takes a single fluorescence frame (float32 numpy array, shape

(H, W), values in [0, 1]) and returns a perturbed frame (P2/P3/P4/P6), a

perturbed nominal-coordinate array (P1), or both an updated frame and metadata

(P5). Parameter grids below are copied verbatim from App. I.

P5 (single-site rearrangement) is the one operator App. I describes only at

the physical level ("its underlying ion fluorescence center is shifted...")

without a pixel-level algorithm. The implementation here is therefore an

engineering approximation: it cuts a small patch around the site's current

location, replaces the vacated region with a local background estimate, and

re-pastes the patch at the shifted location. This is flagged explicitly

rather than presented as an unambiguous reproduction of the paper.

"""

import numpy as np

import cv2


P1_LEVELS = (0.0, 0.5, 1.0, 2.0)

P2_LEVELS = (0.0, 0.5, 1.0, 1.5)

P3_LEVELS = (1.0, 0.7, 0.5, 0.3)

P4_LEVELS = (0.0, 0.001, 0.005, 0.01)

P5_LEVELS = (0.0, 1.0, 2.0, 3.0)

P6_LEVELS = (1.0, 1.5, 2.0, 3.0)


def _odd_kernel_size(sigma, truncate=4.0):

    radius = max(1, int(truncate * sigma + 0.5))

    return 2 * radius + 1


def p1_coordinate_drift(site_coords, sigma_drift, rng=None):

    """P1: c0_k -> c0_k + eps_k, eps_k ~ N(0, sigma_drift^2 I_2), independent

    per site and per perturbation draw. `site_coords` is (K,2) in (x,y) pixels."""

    site_coords = np.asarray(site_coords, dtype=np.float32)

    if sigma_drift <= 0:

        return site_coords.copy()

    rng = rng or np.random.default_rng()

    noise = rng.normal(0.0, sigma_drift, size=site_coords.shape).astype(np.float32)

    return site_coords + noise


def p2_psf_broadening(image, sigma_blur):

    """P2: convolve with an additional isotropic Gaussian of width sigma_blur."""

    image = np.asarray(image, dtype=np.float32)

    if sigma_blur <= 0:

        return image.copy()

    k = _odd_kernel_size(sigma_blur)

    return cv2.GaussianBlur(image, (k, k), sigmaX=sigma_blur, sigmaY=sigma_blur)


def p3_snr_reduction(image, rho, bg_percentile=10.0, rng=None):

    """P3: split into foreground/background via a percentile background

    estimate, scale the foreground by rho, and add Gaussian noise of variance

    (1-rho)*Var(I_fg) so the marginal intensity range matches the unperturbed

    frame."""

    image = np.asarray(image, dtype=np.float32)

    if rho >= 1.0:

        return image.copy()

    rng = rng or np.random.default_rng()

    b_est = float(np.percentile(image, bg_percentile))

    fg = np.clip(image - b_est, 0.0, None)

    background = image - fg

    fg_var = float(fg.var())

    noise_std = float(np.sqrt(max((1.0 - rho) * fg_var, 0.0)))

    noise = rng.normal(0.0, noise_std, size=image.shape).astype(np.float32) if noise_std > 0 else 0.0

    out = background + rho * fg + noise

    return np.clip(out, 0.0, 1.0)


def p4_hot_pixel_injection(image, eta, saturation_value=1.0, rng=None):

    """P4: a uniformly random fraction eta of pixels is set to the sensor's

    saturation value; the same hot-pixel set is used within one draw."""

    image = np.asarray(image, dtype=np.float32).copy()

    if eta <= 0:

        return image

    rng = rng or np.random.default_rng()

    num_pixels = image.size

    num_hot = int(round(eta * num_pixels))

    if num_hot <= 0:

        return image

    flat_idx = rng.choice(num_pixels, size=num_hot, replace=False)

    flat = image.reshape(-1)

    flat[flat_idx] = saturation_value

    return flat.reshape(image.shape)


def p5_single_site_rearrangement(image, site_coords, valid, delta_r, patch_radius=6, rng=None):

    """P5 (approximation, see module docstring): shifts one valid site's local

    fluorescence patch by delta_r pixels in a random direction; the nominal

    lattice supplied to the model is left unchanged by the caller (this

    function only edits the image). Returns (perturbed_image, meta) where meta

    is None if no shift was applied (delta_r<=0 or no valid site) and

    otherwise a dict with the shifted site index and offset.

    """

    image = np.asarray(image, dtype=np.float32).copy()

    if delta_r <= 0:

        return image, None

    valid = np.asarray(valid)

    valid_idx = np.flatnonzero(valid > 0)

    if valid_idx.size == 0:

        return image, None


    rng = rng or np.random.default_rng()

    site_idx = int(rng.choice(valid_idx))

    cx, cy = site_coords[site_idx]

    r = patch_radius

    bg_est = float(np.percentile(image, 10.0))


    padded = np.pad(image, r, mode="constant", constant_values=bg_est)

    pcy, pcx = int(round(cy)) + r, int(round(cx)) + r

    patch = padded[pcy - r : pcy + r + 1, pcx - r : pcx + r + 1].copy()

    padded[pcy - r : pcy + r + 1, pcx - r : pcx + r + 1] = bg_est


    angle = rng.uniform(0, 2 * np.pi)

    dx = delta_r * np.cos(angle)

    dy = delta_r * np.sin(angle)

    new_cy = int(np.clip(round(cy + dy) + r, r, padded.shape[0] - r - 1))

    new_cx = int(np.clip(round(cx + dx) + r, r, padded.shape[1] - r - 1))


    padded[new_cy - r : new_cy + r + 1, new_cx - r : new_cx + r + 1] = np.maximum(

        padded[new_cy - r : new_cy + r + 1, new_cx - r : new_cx + r + 1], patch

    )


    out = padded[r:-r, r:-r]

    meta = {"site_index": site_idx, "shift_dx": float(dx), "shift_dy": float(dy)}

    return out, meta


def p6_anisotropic_psf_elongation(image, ratio, sigma_x=1.0):

    """P6: asymmetric 2D Gaussian blur; minor-axis width sigma_x is fixed,

    major-axis width is sigma_y = ratio * sigma_x."""

    image = np.asarray(image, dtype=np.float32)

    sigma_y = ratio * sigma_x

    kx = _odd_kernel_size(sigma_x)

    ky = _odd_kernel_size(sigma_y)

    return cv2.GaussianBlur(image, (kx, ky), sigmaX=sigma_x, sigmaY=sigma_y)


PERTURBATION_LEVELS = {

    "P1": P1_LEVELS,

    "P2": P2_LEVELS,

    "P3": P3_LEVELS,

    "P4": P4_LEVELS,

    "P5": P5_LEVELS,

    "P6": P6_LEVELS,

}

