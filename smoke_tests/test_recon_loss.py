"""Shape/gradient-flow smoke test for the PSF reconstruction consistency loss

(loss.py: PSFForwardModel, SiteDIAMultitaskLoss, ReconLossSchedule).

"""

import _pathsetup  # noqa: F401

import torch

from loss import (

    PSFForwardModel,

    ReconLossSchedule,

    SiteDIAMultitaskLoss,

    compute_saturation_threshold,

)


def _make_batch(B, H, W, K):

    image = torch.rand(B, 1, H, W)

    bright_logit = torch.randn(B, K, requires_grad=True)

    pred_coords = (torch.rand(B, K, 2) * torch.tensor([W - 1.0, H - 1.0])).requires_grad_(True)

    exist_logit = torch.randn(B, K, requires_grad=True)

    tok_offset = torch.randn(B, K, 2, requires_grad=True)

    outputs = {

        "mask_logits": torch.zeros(B, 1, H, W),

        "bright_logit": bright_logit,

        "pred_coords": pred_coords,

        "exist_logit": exist_logit,

        "dia_tok_offset": tok_offset,

    }

    batch = {

        "image": image,

        "mask": torch.zeros(B, 1, H, W),

        "state_hard": (torch.rand(B, K) > 0.5).float(),

        "valid": torch.ones(B, K),

        "site_coords": pred_coords.detach(),

    }

    return outputs, batch


def test_recon_loss_schedule():

    sched = ReconLossSchedule(warm_epochs=2, ramp_epochs=2, max_weight=0.05)

    assert sched.weight_at(0) == 0.0

    assert sched.detach_state_grad_at(0) is True

    assert 0.0 < sched.weight_at(3) < 0.05

    assert sched.weight_at(10) == 0.05

    assert sched.detach_state_grad_at(10) is False


    sched_no_warmup = ReconLossSchedule(use_warmup=False, max_weight=0.05)

    assert sched_no_warmup.weight_at(0) == 0.05

    assert sched_no_warmup.detach_state_grad_at(0) is False


    sched_disabled = ReconLossSchedule(disabled=True, max_weight=0.05)

    assert sched_disabled.weight_at(50) == 0.0


def test_psf_forward_and_gradients():

    torch.manual_seed(0)

    B, H, W, K = 2, 40, 60, 15


    outputs, batch = _make_batch(B, H, W, K)

    t_sat = compute_saturation_threshold([batch["image"][i, 0] for i in range(B)])

    assert 0.0 <= t_sat <= 1.0


    psf_model = PSFForwardModel(sigma_init=1.5, recon_radius=6)

    criterion = SiteDIAMultitaskLoss(mask_weight=0.0, psf_model=psf_model, t_sat=t_sat)


    out = criterion(outputs, batch, recon_weight=0.05, detach_state_grad=False)

    assert "loss_recon" in out

    out["loss"].backward()


    assert psf_model.log_sigma.grad is not None

    assert psf_model.b.grad is not None

    assert psf_model.s.grad is not None

    assert outputs["pred_coords"].grad is not None


def test_detach_state_grad_blocks_bright_logit():

    torch.manual_seed(0)

    B, H, W, K = 2, 40, 60, 15

    outputs, batch = _make_batch(B, H, W, K)

    t_sat = compute_saturation_threshold([batch["image"][i, 0] for i in range(B)])


    psf_model = PSFForwardModel(sigma_init=1.5, recon_radius=6)

    criterion = SiteDIAMultitaskLoss(

        mask_weight=0.0, state_weight=0.0, coord_weight=0.0, exist_weight=0.0,

        offset_reg_weight=0.0, psf_model=psf_model, t_sat=t_sat,

    )

    out = criterion(outputs, batch, recon_weight=0.05, detach_state_grad=True)

    out["loss"].backward()

    assert outputs["bright_logit"].grad is None or outputs["bright_logit"].grad.abs().sum() == 0


def test_missing_exist_logit_is_tolerated():

    """Coordinate-aware baselines (App. F) have no existence head."""

    torch.manual_seed(0)

    B, H, W, K = 1, 30, 30, 8

    outputs, batch = _make_batch(B, H, W, K)

    del outputs["exist_logit"]

    del outputs["dia_tok_offset"]

    criterion = SiteDIAMultitaskLoss(mask_weight=0.0)

    out = criterion(outputs, batch)

    out["loss"].backward()


if __name__ == "__main__":

    test_recon_loss_schedule()

    test_psf_forward_and_gradients()

    test_detach_state_grad_blocks_bright_logit()

    test_missing_exist_logit_is_tolerated()

    print("PASS: test_recon_loss")

