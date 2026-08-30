"""Converts a trained field-dependent PSF kernel into physical quantities.


The eleven raw parameters of FieldDependentKernel are stored in the

parameterizations that keep the covariance positive definite and the

amplitudes bounded, which makes them unreadable on their own. This script

maps them back to quantities that can be checked against the optics and the

trap model independently of the readout metrics:


    * the fitted field center, in absolute pixels

    * the base profile widths along its two principal axes, in pixels, and the

      orientation of the major axis

    * the Moffat beta, with the profile value it implies at four and seven

      pixels, directly comparable to the measured 12% and 1.6% shoulders

    * the coma amplitude, expressed as the intensity ratio it produces between

      the two sides of a site at the edge of the register

    * the astigmatism variances at the edge of the register, in pixels, as the

      extra width they add along the radial and tangential axes

    * the micromotion amplitude at the edge of the register, in pixels, and the

      orientation of its axis


Usage:


    python report_psf_parameters.py --checkpoint outputs/site_dia/best.pt \

        --frame-height 88 --frame-width 456

"""


import argparse

import math


import torch


from psf_kernels import FieldDependentKernel


def _load_kernel_state(checkpoint_path):

    """Pulls the kernel parameters out of a training checkpoint.


    Accepts either a bare state_dict or a checkpoint dict that carries one

    under a common key, and tolerates the 'psf_model.' prefix that appears

    when the renderer is saved as part of a larger module.

    """

    blob = torch.load(checkpoint_path, map_location="cpu", weights_only=False)


    state = blob

    if isinstance(blob, dict):

        for key in ("psf_model", "state_dict", "model_state_dict", "model"):

            if key in blob and isinstance(blob[key], dict):

                state = blob[key]

                break


    kernel_state = {}

    for name, value in state.items():

        if not isinstance(value, torch.Tensor):

            continue

        short = name.split("kernel.")[-1] if "kernel." in name else name

        short = short.split("psf_model.")[-1]

        if short.startswith("raw_"):

            kernel_state[short] = value


    missing = {"raw_chol", "raw_beta", "raw_center", "raw_coma", "raw_astig",

               "raw_micromotion"} - set(kernel_state)

    if missing:

        raise SystemExit(

            f"{checkpoint_path} does not contain a field-dependent PSF kernel "
            f"(missing {sorted(missing)}). Was it trained with "
            f"--site_psf_kernel field_dependent?"

        )

    return kernel_state


@torch.no_grad()

def report(kernel, frame_shape):

    H, W = frame_shape

    half_diagonal = 0.5 * math.sqrt((W - 1) ** 2 + (H - 1) ** 2)


    a, b, c = (float(x) for x in kernel.raw_chol)

    s00 = math.exp(2 * a)

    s01 = math.exp(a) * b

    s11 = b**2 + math.exp(2 * c)

    cov = torch.tensor([[s00, s01], [s01, s11]])

    eigvals, eigvecs = torch.linalg.eigh(cov)

    widths = eigvals.clamp(min=1e-12).sqrt()

    major = eigvecs[:, int(torch.argmax(eigvals))]

    orientation = math.degrees(math.atan2(float(major[1]), float(major[0])))


    beta = float(torch.nn.functional.softplus(kernel.raw_beta) + 1.0)

    sigma_eff = float(widths.mean())

    def moffat_at(radius_px):

        q = (radius_px / sigma_eff) ** 2

        return (1.0 + q / (2.0 * beta)) ** (-beta)


    center_x = (W - 1) / 2.0 + float(kernel.raw_center[0])

    center_y = (H - 1) / 2.0 + float(kernel.raw_center[1])


    coma = kernel.coma_scale * math.tanh(float(kernel.raw_coma))

    # At the far edge the normalized field radius is 1 by construction, so the
    # tilt across +-4 px is 8 * coma and the two sides differ by exp of that.

    edge_ratio = math.exp(min(8.0 * coma, 8.0))


    astig = torch.nn.functional.softplus(kernel.raw_astig)

    var_radial = float(astig[0])

    var_tangential = float(astig[1])


    amp = float(torch.nn.functional.softplus(kernel.raw_micromotion[0]))

    angle = math.degrees(float(kernel.raw_micromotion[1])) % 180.0


    print(f"frame                        {W} x {H} px, half-diagonal {half_diagonal:.1f} px")

    print()

    print("field center")

    print(f"  fitted c0                  ({center_x:.2f}, {center_y:.2f}) px")

    print(f"  offset from frame center   ({float(kernel.raw_center[0]):+.2f}, "

          f"{float(kernel.raw_center[1]):+.2f}) px")

    print()

    print("base profile (field independent)")

    print(f"  principal widths           {float(widths[0]):.3f} / {float(widths[1]):.3f} px")

    print(f"  major-axis orientation     {orientation:+.1f} deg")

    print(f"  Moffat beta                {beta:.3f}   (Gaussian limit: beta -> inf)")

    print(f"  profile at 4 px            {moffat_at(4.0)*100:.2f}% of peak")

    print(f"  profile at 7 px            {moffat_at(7.0)*100:.2f}% of peak")

    print()

    print("field-dependent terms, evaluated at the edge of the register")

    print(f"  coma coefficient           {coma:+.4f}")

    print(f"  implied +-4 px side ratio  {edge_ratio:.2f}   (1.00 means no coma)")

    print(f"  astigmatism radial         {math.sqrt(var_radial):.3f} px added in quadrature")

    print(f"  astigmatism tangential     {math.sqrt(var_tangential):.3f} px added in quadrature")

    print(f"  micromotion amplitude      {amp:.3f} px")

    print(f"  micromotion axis           {angle:.1f} deg from the long frame axis")


def main():

    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument("--checkpoint", required=True, help="Trained checkpoint containing the field-dependent PSF kernel.")

    parser.add_argument("--frame-height", type=int, default=88, help="Frame height in pixels; sets the field normalization.")

    parser.add_argument("--frame-width", type=int, default=456, help="Frame width in pixels; sets the field normalization.")

    args = parser.parse_args()


    kernel_state = _load_kernel_state(args.checkpoint)

    kernel = FieldDependentKernel()

    with torch.no_grad():

        for name, value in kernel_state.items():

            getattr(kernel, name).copy_(value)


    report(kernel, (args.frame_height, args.frame_width))


if __name__ == "__main__":

    main()
