import torch

import torch.nn.functional as F


def gaussian_window(window_size=11, sigma=1.5, channels=1, device="cpu", dtype=torch.float32):

    coords = torch.arange(window_size, device=device, dtype=dtype) - window_size // 2

    gauss = torch.exp(-(coords**2) / (2 * sigma**2))

    gauss = gauss / gauss.sum()

    kernel_2d = torch.outer(gauss, gauss)

    kernel_2d = kernel_2d / kernel_2d.sum()

    return kernel_2d.view(1, 1, window_size, window_size).repeat(channels, 1, 1, 1)


def ssim_map(pred, target, window_size=11, sigma=1.5, c1=0.01**2, c2=0.03**2):

    """Local SSIM map, same spatial size as the inputs ('same' padding)."""

    channels = pred.size(1)

    window = gaussian_window(window_size, sigma, channels, pred.device, pred.dtype)


    mu_x = F.conv2d(pred, window, padding=window_size // 2, groups=channels)

    mu_y = F.conv2d(target, window, padding=window_size // 2, groups=channels)


    mu_x_sq = mu_x.pow(2)

    mu_y_sq = mu_y.pow(2)

    mu_xy = mu_x * mu_y


    sigma_x_sq = (

        F.conv2d(pred * pred, window, padding=window_size // 2, groups=channels) - mu_x_sq

    )

    sigma_y_sq = (

        F.conv2d(target * target, window, padding=window_size // 2, groups=channels) - mu_y_sq

    )

    sigma_xy = (

        F.conv2d(pred * target, window, padding=window_size // 2, groups=channels) - mu_xy

    )


    numerator = (2 * mu_xy + c1) * (2 * sigma_xy + c2)

    denominator = (mu_x_sq + mu_y_sq + c1) * (sigma_x_sq + sigma_y_sq + c2)

    return numerator / (denominator + 1e-8)


def ssim_loss(pred, target, window_size=11, sigma=1.5, c1=0.01**2, c2=0.03**2):

    return 1.0 - ssim_map(pred, target, window_size, sigma, c1, c2).mean()


def masked_ssim_loss(

    pred, target, valid_mask, window_size=11, sigma=1.5, c1=0.01**2, c2=0.03**2, eps=1e-6

):

    """Averages the local SSIM map only over windows whose pixels are ALL valid.

    Matches the Site-AIT paper's Eq. 9: saturated pixels contribute to neither the

    pixelwise nor the structural component of the reconstruction loss, because

    SSIM_{m_i} is only averaged over windows fully contained in the non-saturated

    region m_i(x)=1.

    """

    smap = ssim_map(pred, target, window_size, sigma, c1, c2)

    channels = pred.size(1)

    box = torch.ones(1, 1, window_size, window_size, device=pred.device, dtype=pred.dtype).repeat(

        channels, 1, 1, 1

    )

    valid_frac = F.conv2d(

        valid_mask.to(pred.dtype), box, padding=window_size // 2, groups=channels

    ) / (window_size * window_size)

    window_valid = (valid_frac >= 1.0 - 1e-6).to(smap.dtype)

    denom = window_valid.sum().clamp_min(eps)

    masked_mean = (smap * window_valid).sum() / denom

    return 1.0 - masked_mean

