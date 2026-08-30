"""Smoke test for the PSF kernels (psf_kernels.py) used by the reconstruction

consistency loss: the published isotropic Gaussian, the anisotropic Moffat, the

free-form empirical grid, and the field-dependent kernel.

"""

import _pathsetup  # noqa: F401

import math

import torch

from loss import PSFForwardModel

from psf_kernels import (

    AnisotropicMoffatKernel,

    EmpiricalGridKernel,

    FieldDependentKernel,

    build_psf_kernel,

)


KERNELS = ("isotropic", "anisotropic", "empirical", "field_dependent")


def _render(kernel_name, **kwargs):

    torch.manual_seed(0)

    B, K, H, W = 2, 6, 40, 60

    model = PSFForwardModel(sigma_init=1.36, recon_radius=6, kernel=kernel_name, **kwargs)

    pred_p = torch.rand(B, K)

    coords = (torch.rand(B, K, 2) * torch.tensor([W - 1.0, H - 1.0])).requires_grad_(True)

    valid = torch.ones(B, K)

    image = model(pred_p, coords, valid, (H, W))

    return model, coords, image


def test_all_kernels_render_and_backprop():

    """Backpropagates the L1 term of L_recon rather than image.sum().


    image.sum() is degenerate here: the non-analytic kernels are normalized over

    the full support, so the total splatted flux is 1 per site whatever the

    profile shape and every shape parameter receives exactly zero gradient. Only

    a pixelwise objective exercises them.

    """

    torch.manual_seed(0)

    target = torch.rand(2, 1, 40, 60)


    for name in KERNELS:

        model, coords, image = _render(name)

        assert image.shape == (2, 1, 40, 60)

        assert torch.isfinite(image).all()

        (image - target).abs().mean().backward()


        assert coords.grad is not None

        assert coords.grad.abs().sum() > 0, f"{name}: no gradient to pred_coords"

        assert model.b.grad is not None

        assert model.s.grad is not None


        for param_name, param in model.kernel.named_parameters():

            assert param.grad is not None, f"{name}: {param_name} received no gradient"

            assert param.grad.abs().sum() > 0, f"{name}: {param_name} has zero gradient"


def test_isotropic_is_unchanged_and_checkpoint_compatible():

    """The published configuration must stay bit-identical, and a checkpoint

    written before the kernels were added must still load.

    """

    model, _, _ = _render("isotropic")

    assert set(model.state_dict()) == {"log_sigma", "b", "s"}


    legacy = {

        "log_sigma": torch.tensor(0.3),

        "b": torch.tensor(0.01),

        "s": torch.tensor(2.0),

    }

    model.load_state_dict(legacy, strict=True)


def test_new_kernels_carry_shape_parameters():

    aniso = PSFForwardModel(recon_radius=6, kernel="anisotropic")

    assert sum(p.numel() for p in aniso.kernel.parameters()) == 6


    grid = PSFForwardModel(recon_radius=6, kernel="empirical")

    assert sum(p.numel() for p in grid.kernel.parameters()) == 13 * 13


def test_freeze_shape_covers_every_shape_parameter():

    for name in KERNELS:

        model = PSFForwardModel(recon_radius=6, kernel=name)

        model.freeze_shape()

        assert not model.log_sigma.requires_grad

        for param in model.kernel.parameters():

            assert not param.requires_grad, f"{name}: shape parameter left trainable"

        # The photometric scale must stay trainable.

        assert model.b.requires_grad and model.s.requires_grad


def test_non_analytic_kernels_are_flux_normalized():

    """A site far from any edge must deposit the same total flux whatever the

    profile shape, so that the learned photon scale s keeps its meaning.

    """

    for name in ("anisotropic", "empirical", "field_dependent"):

        model = PSFForwardModel(sigma_init=1.36, recon_radius=6, kernel=name)

        with torch.no_grad():

            model.b.zero_()

            model.s.fill_(1.0)

        coords = torch.tensor([[[30.0, 20.0]]])

        image = model(torch.ones(1, 1), coords, torch.ones(1, 1), (40, 60))

        assert math.isclose(image.sum().item(), 1.0, rel_tol=1e-4), name


def test_empirical_profile_stays_non_negative():

    kernel = EmpiricalGridKernel(recon_radius=6, sigma_init=1.36)

    with torch.no_grad():

        kernel.raw_grid.add_(torch.randn_like(kernel.raw_grid) * 5.0)

    assert (kernel.profile >= 0).all()


def test_moffat_is_heavier_tailed_than_its_gaussian_limit():

    """Sec. II.B measures a profile far above the fitted Gaussian at 4 px. A

    finite beta must reproduce that direction; large beta must fall back to the

    Gaussian.

    """

    offset = torch.tensor([[4.0, 0.0]])

    origin = torch.zeros(1, 2)


    heavy = AnisotropicMoffatKernel(sigma_init=1.36, beta_init=1.6)

    light = AnisotropicMoffatKernel(sigma_init=1.36, beta_init=200.0)


    heavy_ratio = (heavy(offset) / heavy(origin)).item()

    light_ratio = (light(offset) / light(origin)).item()

    gaussian_ratio = math.exp(-((4.0 / 1.36) ** 2) / 2)


    assert heavy_ratio > light_ratio

    assert math.isclose(light_ratio, gaussian_ratio, rel_tol=0.05)


def test_field_dependent_parameter_budget():

    """Eleven shape numbers cover the whole register, against 169 for a single

    free grid and 300x that for one kernel per site.

    """

    model = PSFForwardModel(recon_radius=6, kernel="field_dependent")

    assert sum(p.numel() for p in model.kernel.parameters()) == 11


def test_field_dependence_starts_negligible():

    """An untrained field_dependent model must render essentially the same

    profile everywhere, so training starts from the field-independent solution.


    Coma starts at exactly zero, but the two variance amplitudes cannot: softplus

    is what keeps the covariance positive definite and its gradient dies as its

    output goes to zero. They start at 0.049 px^2 against a base sigma^2 of

    1.85 px^2, which leaves a sub-percent difference between the center and the

    edge of the register.

    """

    kernel = FieldDependentKernel(sigma_init=1.36)

    shape = (88, 456)

    offsets = torch.tensor([[[[float(dx), 0.0] for dx in range(-6, 7)]]])


    center = kernel(offsets, coords=torch.tensor([[[227.5, 43.5]]]), image_shape=shape)

    edge = kernel(offsets, coords=torch.tensor([[[455.0, 43.5]]]), image_shape=shape)


    relative = ((edge - center).abs() / center.clamp(min=1e-12)).max().item()

    assert relative < 0.05, relative


def test_astigmatism_broadens_rather_than_narrows():

    """Regression test for a sign error. The field terms are variances and must

    compose in covariance space; adding them to the precision matrix instead

    makes an aberration that should broaden the image narrow it.

    """

    kernel = FieldDependentKernel(sigma_init=1.36)

    with torch.no_grad():

        kernel.raw_astig.copy_(torch.tensor([2.0, -3.0]))


    shape = (88, 456)

    offsets = torch.tensor([[[[float(dx), 0.0] for dx in range(-6, 7)]]])


    def width(x):

        v = kernel(offsets, coords=torch.tensor([[[x, 43.5]]]), image_shape=shape)

        return (v / v.max()).sum().item()


    assert width(455.0) > width(227.5)


def test_coma_vanishes_on_axis_and_grows_with_field_radius():

    """Aberration theory requires coma to be exactly zero at the field center

    and to grow linearly with field radius. That is the signature separating a

    field-dependent model from a single shared kernel.

    """

    kernel = FieldDependentKernel(sigma_init=1.36)

    with torch.no_grad():

        kernel.raw_coma.fill_(1.2)


    shape = (88, 456)

    offsets = torch.tensor([[[[4.0, 0.0], [-4.0, 0.0]]]])


    def asymmetry(x):

        v = kernel(offsets, coords=torch.tensor([[[x, 43.5]]]), image_shape=shape)

        return (v[0, 0, 0] / v[0, 0, 1]).item()


    on_axis = asymmetry(227.5)

    mid = asymmetry(341.0)

    edge = asymmetry(455.0)


    assert math.isclose(on_axis, 1.0, rel_tol=1e-3), on_axis

    assert edge > mid > on_axis


def test_field_shape_does_not_leak_into_the_coordinate_gradient():

    """The profile at a field point is a property of the optics. The coordinate

    head must not be able to move a site in order to obtain a kernel that fits

    the frame better, so the field geometry is computed on detached coordinates.

    """

    kernel = FieldDependentKernel(sigma_init=1.36)

    assert kernel.detach_field_coords


    with torch.no_grad():

        kernel.raw_coma.fill_(1.2)


    coords = torch.tensor([[[455.0, 43.5]]], requires_grad=True)

    offsets = torch.zeros(1, 1, 1, 2)

    kernel(offsets, coords=coords, image_shape=(88, 456)).sum().backward()

    # The offsets are identically zero, so the only path to coords would be the
    # field-dependent shape; it must be cut.

    assert coords.grad is None or coords.grad.abs().sum() == 0


def test_build_psf_kernel_rejects_unknown_name():

    try:

        build_psf_kernel("gaussian")

    except ValueError:

        return

    raise AssertionError("expected ValueError for an unknown kernel name")


if __name__ == "__main__":

    test_all_kernels_render_and_backprop()

    test_isotropic_is_unchanged_and_checkpoint_compatible()

    test_new_kernels_carry_shape_parameters()

    test_freeze_shape_covers_every_shape_parameter()

    test_non_analytic_kernels_are_flux_normalized()

    test_empirical_profile_stays_non_negative()

    test_moffat_is_heavier_tailed_than_its_gaussian_limit()

    test_field_dependent_parameter_budget()

    test_field_dependence_starts_negligible()

    test_astigmatism_broadens_rather_than_narrows()

    test_coma_vanishes_on_axis_and_grows_with_field_radius()

    test_field_shape_does_not_leak_into_the_coordinate_gradient()

    test_build_psf_kernel_rejects_unknown_name()

    print("PASS: test_psf_kernels")
