import math

import torch

import torch.nn as nn

import torch.nn.functional as F

from ssim_utils import masked_ssim_loss

from psf_kernels import build_psf_kernel


def dice_loss(pred, target, eps=1.0):


    pred = pred.contiguous().view(pred.size(0), -1)

    target = target.contiguous().view(target.size(0), -1)


    intersection = (pred * target).sum(dim=1)

    union = pred.sum(dim=1) + target.sum(dim=1)


    loss = 1.0 - (2.0 * intersection + eps) / (union + eps)

    return loss.mean()


def extract_local_soft_centroid(

    prob_map, centers_gt, centers_valid, radius=4, eps=1e-6

):


    B, C, H, W = prob_map.shape

    assert C == 1


    device = prob_map.device

    dtype = prob_map.dtype

    K = centers_gt.shape[1]


    ys_full = torch.arange(H, device=device, dtype=dtype).view(1, 1, H, 1)

    xs_full = torch.arange(W, device=device, dtype=dtype).view(1, 1, 1, W)


    pred_centers = torch.zeros((B, K, 2), device=device, dtype=dtype)


    for b in range(B):

        for k in range(K):

            if centers_valid[b, k] < 0.5:

                continue


            x0 = centers_gt[b, k, 0]

            y0 = centers_gt[b, k, 1]


            x_min = max(0, int(torch.floor(x0).item()) - radius)

            x_max = min(W, int(torch.floor(x0).item()) + radius + 1)

            y_min = max(0, int(torch.floor(y0).item()) - radius)

            y_max = min(H, int(torch.floor(y0).item()) + radius + 1)


            patch = prob_map[b : b + 1, :, y_min:y_max, x_min:x_max]

            if patch.numel() == 0:

                pred_centers[b, k] = centers_gt[b, k]

                continue


            xs = xs_full[:, :, :, x_min:x_max].expand_as(patch)

            ys = ys_full[:, :, y_min:y_max, :].expand_as(patch)


            mass = patch.sum() + eps

            cx = (patch * xs).sum() / mass

            cy = (patch * ys).sum() / mass


            pred_centers[b, k, 0] = cx

            pred_centers[b, k, 1] = cy


    return pred_centers


def multi_ion_centroid_loss(pred, centers_gt, centers_valid, radius=4):


    pred_centers = extract_local_soft_centroid(

        pred, centers_gt, centers_valid, radius=radius

    )


    diff = (pred_centers - centers_gt) ** 2

    diff = diff.sum(dim=-1)


    valid = centers_valid.float()

    loss = (diff * valid).sum() / (valid.sum() + 1e-6)

    return loss


class HybridSegmentationMultiIonLoss(nn.Module):


    def __init__(self, bce_weight=1.0, dice_weight=1.0, centroid_weight=0.1, radius=4):

        super().__init__()

        self.bce_weight = bce_weight

        self.dice_weight = dice_weight

        self.centroid_weight = centroid_weight

        self.radius = radius

        self.bce = nn.BCEWithLogitsLoss()


    def forward(self, logits, target, centers_gt, centers_valid):

        pred = torch.sigmoid(logits)


        loss_bce = self.bce(logits, target)

        loss_dice = dice_loss(pred, target)

        if self.centroid_weight > 0:

            loss_centroid = multi_ion_centroid_loss(

                pred, centers_gt, centers_valid, radius=self.radius

            )

        else:

            loss_centroid = torch.zeros((), device=logits.device, dtype=logits.dtype)


        total_loss = (

            self.bce_weight * loss_bce

            + self.dice_weight * loss_dice

            + self.centroid_weight * loss_centroid

        )


        return {

            "loss": total_loss,

            "loss_bce": loss_bce,

            "loss_dice": loss_dice,

            "loss_centroid": loss_centroid,

        }


def masked_mean(loss, valid, eps=1e-6):

    valid = valid.float()

    return (loss * valid).sum() / (valid.sum() + eps)


class PSFForwardModel(nn.Module):

    """Differentiable PSF forward model for the physics-consistent training

    objective (Site-AIT paper Sec. IV.G, Eq. 6-7): synthesizes the fluorescence

    frame predicted by the current state/coordinate estimates, as a sparse splat

    of K kernels evaluated only within a clipping radius R_recon around each

    predicted site coordinate.


    `kernel` selects the profile. 'isotropic' is the original Gaussian and is

    kept as the default so previously trained checkpoints load unchanged.

    'anisotropic' and 'empirical' add the heavy tails and the side-to-side

    asymmetry that Sec. II.B measures on this imaging system; see

    psf_kernels.py.

    """


    def __init__(

        self,

        sigma_init=1.5,

        recon_radius=6,

        kernel="isotropic",

        beta_init=2.0,

        init_profile=None,

    ):

        super().__init__()

        self.log_sigma = nn.Parameter(torch.log(torch.tensor(float(sigma_init))))

        self.b = nn.Parameter(torch.zeros(()))

        self.s = nn.Parameter(torch.ones(()))

        self.recon_radius = recon_radius

        self.kernel_name = kernel

        self.kernel = build_psf_kernel(

            kernel,

            sigma_init=sigma_init,

            recon_radius=recon_radius,

            beta_init=beta_init,

            init_profile=init_profile,

        )


        offsets = torch.stack(

            torch.meshgrid(

                torch.arange(-recon_radius, recon_radius + 1, dtype=torch.float32),

                torch.arange(-recon_radius, recon_radius + 1, dtype=torch.float32),

                indexing="ij",

            ),

            dim=-1,

        ).reshape(-1, 2)

        self.register_buffer("_offsets_rc", offsets, persistent=False)


    @property

    def sigma_psf(self):

        return self.log_sigma.exp().clamp(min=0.05)


    def shape_parameters(self):

        """The parameters describing the profile shape, as opposed to the

        photometric scale (b, s). Used by the 'PSF shape frozen' ablation.

        """

        yield self.log_sigma

        for param in self.kernel.parameters():

            yield param


    def freeze_shape(self):

        """Stop optimizing the profile shape. Generalizes the original

        `log_sigma.requires_grad_(False)` to the anisotropic and empirical

        kernels, which carry additional shape parameters.

        """

        for param in self.shape_parameters():

            param.requires_grad_(False)


    def forward(self, pred_p, pred_coords, valid, image_shape):

        """

        pred_p: (B,K) bright-state probabilities to splat (caller applies

            .detach() beforehand when the state-head gradient must be blocked).

        pred_coords: (B,K,2) predicted (x,y) pixel coordinates.

        valid: (B,K) site-validity gate v_{i,k}.

        image_shape: (H, W) of the target frame.

        Returns the synthesized image, shape (B,1,H,W).

        """

        H, W = image_shape

        B, K = pred_p.shape

        device = pred_coords.device

        dtype = pred_coords.dtype

        sigma = self.sigma_psf.to(dtype=dtype)


        center_x = pred_coords[..., 0].round()

        center_y = pred_coords[..., 1].round()


        dxy = torch.stack(

            [self._offsets_rc[:, 1], self._offsets_rc[:, 0]], dim=-1

        ).to(device=device, dtype=dtype)


        px = center_x.unsqueeze(-1) + dxy[:, 0].view(1, 1, -1)

        py = center_y.unsqueeze(-1) + dxy[:, 1].view(1, 1, -1)


        in_bounds = (px >= 0) & (px <= W - 1) & (py >= 0) & (py <= H - 1)

        px_c = px.clamp(0, W - 1)

        py_c = py.clamp(0, H - 1)


        delta = torch.stack(

            [px - pred_coords[..., 0:1], py - pred_coords[..., 1:2]], dim=-1

        )

        # pred_coords and the frame shape are passed through because the
        # field-dependent kernel varies its profile across the register; the
        # other kernels ignore them.

        psi = self.kernel(delta, sigma, coords=pred_coords, image_shape=image_shape)


        if not self.kernel.analytic_norm:

            # Normalize over the full support and before the in-bounds mask, so

            # that a site whose support is clipped by the frame edge loses that

            # fraction of its flux instead of redistributing it inward.

            psi = psi / psi.sum(dim=-1, keepdim=True).clamp(min=1e-12)


        weight = valid.unsqueeze(-1) * pred_p.unsqueeze(-1) * psi

        weight = weight * in_bounds.to(dtype)


        flat_idx = (py_c.long() * W + px_c.long()).view(B, -1)

        weight = weight.reshape(B, -1)


        image = torch.zeros(B, H * W, device=device, dtype=dtype)

        image.scatter_add_(1, flat_idx, weight)

        image = image.view(B, 1, H, W)


        return self.b + self.s * image


def compute_saturation_threshold(images, percentile=99.9):

    """T_sat = Q_{0.999} percentile of training-pixel intensities (Site-AIT paper,

    Sec. IV.G). Meant to be computed once, before training, over the training

    split, then frozen for the entire run. `images` is an iterable of 2D/3D

    tensors or arrays with values in the same [0,1] range as the model input.

    """

    values = []

    for img in images:

        t = torch.as_tensor(img, dtype=torch.float32).flatten()

        values.append(t)

    all_pixels = torch.cat(values)

    return float(torch.quantile(all_pixels, percentile / 100.0))


class ReconLossSchedule:

    """App. H warm-up schedule for lambda_recon and the state-head gradient

    detachment flag:

        lambda_recon(t) = 0                                   for t <  E_warm

                         = lambda_max * (t - E_warm) / E_ramp  for E_warm <= t < E_warm+E_ramp

                         = lambda_max                          otherwise

    and the state-head gradient into the reconstruction term is detached for

    the first E_warm epochs. `use_warmup=False` collapses straight to

    lambda_max with no detachment ('no warm-up' ablation row); `disabled=True`

    always returns weight 0 ('- L_recon, no recon. loss' row); `freeze_sigma`

    is read by the caller to stop optimizing sigma_psf ('sigma_psf frozen' row).

    """


    def __init__(

        self,

        warm_epochs=10,

        ramp_epochs=5,

        max_weight=0.05,

        use_warmup=True,

        freeze_sigma=False,

        disabled=False,

    ):

        self.warm_epochs = warm_epochs

        self.ramp_epochs = ramp_epochs

        self.max_weight = max_weight

        self.use_warmup = use_warmup

        self.freeze_sigma = freeze_sigma

        self.disabled = disabled


    def weight_at(self, epoch):

        if self.disabled:

            return 0.0

        if not self.use_warmup:

            return self.max_weight

        if epoch < self.warm_epochs:

            return 0.0

        if epoch < self.warm_epochs + self.ramp_epochs:

            return self.max_weight * (epoch - self.warm_epochs) / max(self.ramp_epochs, 1)

        return self.max_weight


    def detach_state_grad_at(self, epoch):

        if self.disabled:

            return True

        if not self.use_warmup:

            return False

        return epoch < self.warm_epochs


class SiteDIAMultitaskLoss(nn.Module):


    def __init__(

        self,

        mask_weight=0.2,

        state_weight=1.0,

        coord_weight=0.05,

        exist_weight=0.1,

        offset_reg_weight=0.01,

        dice_weight=1.0,

        bce_weight=1.0,

        psf_model=None,

        t_sat=None,

        recon_lambda_ssim=0.2,

    ):

        super().__init__()

        self.mask_weight = mask_weight

        self.state_weight = state_weight

        self.coord_weight = coord_weight

        self.exist_weight = exist_weight

        self.offset_reg_weight = offset_reg_weight

        self.dice_weight = dice_weight

        self.bce_weight = bce_weight

        self.bce_logits = nn.BCEWithLogitsLoss()

        self.psf_model = psf_model

        self.t_sat = t_sat

        self.recon_lambda_ssim = recon_lambda_ssim


    def forward(self, outputs, batch, recon_weight=0.0, detach_state_grad=False):

        mask_logits = outputs["mask_logits"]

        if self.mask_weight != 0:

            mask_target = batch["mask"].to(mask_logits.device, dtype=mask_logits.dtype)

            mask_prob = torch.sigmoid(mask_logits)

            loss_mask_bce = self.bce_logits(mask_logits, mask_target)

            loss_mask_dice = dice_loss(mask_prob, mask_target)

            loss_mask = self.bce_weight * loss_mask_bce + self.dice_weight * loss_mask_dice

        else:


            loss_mask_bce = mask_logits.new_zeros(())

            loss_mask_dice = mask_logits.new_zeros(())

            loss_mask = mask_logits.new_zeros(())


        state = batch["state_hard"].to(mask_logits.device, dtype=mask_logits.dtype)

        valid = batch["valid"].to(mask_logits.device, dtype=mask_logits.dtype)

        bright_logit = outputs["bright_logit"]

        loss_state_raw = F.binary_cross_entropy_with_logits(

            bright_logit, state, reduction="none"

        )

        loss_state = masked_mean(loss_state_raw, valid)


        target_coords = batch["site_coords"].to(mask_logits.device, dtype=mask_logits.dtype)

        pred_coords = outputs["pred_coords"]

        coord_raw = F.smooth_l1_loss(pred_coords, target_coords, reduction="none").sum(dim=-1)

        loss_coord = masked_mean(coord_raw, valid)


        exist_target = valid.clamp(0, 1)

        exist_logit = outputs.get("exist_logit")

        if exist_logit is not None:

            exist_raw = F.binary_cross_entropy_with_logits(

                exist_logit, exist_target, reduction="none"

            )

            loss_exist = masked_mean(exist_raw, valid)

        else:

            # Coordinate-aware baselines (App. F) have no existence head.
            loss_exist = torch.zeros((), device=mask_logits.device, dtype=mask_logits.dtype)


        tok_offset = outputs.get("dia_tok_offset")

        if tok_offset is not None:

            loss_offset_reg = (tok_offset**2).mean()

        else:

            loss_offset_reg = torch.zeros((), device=mask_logits.device, dtype=mask_logits.dtype)


        if self.psf_model is not None and recon_weight > 0:

            if self.t_sat is None:

                raise ValueError("t_sat must be set (frozen saturation threshold) to use L_recon")

            image = batch["image"].to(mask_logits.device, dtype=mask_logits.dtype)

            pred_p = torch.sigmoid(bright_logit)

            pred_p_for_recon = pred_p.detach() if detach_state_grad else pred_p

            synth = self.psf_model(pred_p_for_recon, pred_coords, valid, image.shape[-2:])

            sat_mask = (image < self.t_sat).to(image.dtype)

            loss_recon_l1 = masked_mean((synth - image).abs(), sat_mask)

            loss_recon_ssim = masked_ssim_loss(synth, image, sat_mask)

            loss_recon = loss_recon_l1 + self.recon_lambda_ssim * loss_recon_ssim

        else:

            loss_recon = torch.zeros((), device=mask_logits.device, dtype=mask_logits.dtype)


        loss = (

            self.mask_weight * loss_mask

            + self.state_weight * loss_state

            + self.coord_weight * loss_coord

            + self.exist_weight * loss_exist

            + self.offset_reg_weight * loss_offset_reg

            + recon_weight * loss_recon

        )


        with torch.no_grad():

            pred_state = (torch.sigmoid(bright_logit) > 0.5).float()

            ion_acc = masked_mean((pred_state == state).float(), valid)

            if exist_logit is not None:

                exist_acc = masked_mean(

                    ((torch.sigmoid(exist_logit) > 0.5).float() == exist_target).float(), valid

                )

            else:

                exist_acc = torch.zeros((), device=mask_logits.device, dtype=mask_logits.dtype)


        return {

            "loss": loss,

            "loss_mask": loss_mask.detach(),

            "loss_mask_bce": loss_mask_bce.detach(),

            "loss_mask_dice": loss_mask_dice.detach(),

            "loss_state": loss_state.detach(),

            "loss_coord": loss_coord.detach(),

            "loss_exist": loss_exist.detach(),

            "loss_offset_reg": loss_offset_reg.detach(),

            "loss_recon": loss_recon.detach(),

            "ion_acc": ion_acc.detach(),

            "exist_acc": exist_acc.detach(),


            "loss_bce": loss_mask_bce.detach(),

            "loss_dice": loss_mask_dice.detach(),

            "loss_centroid": loss_coord.detach(),

        }

