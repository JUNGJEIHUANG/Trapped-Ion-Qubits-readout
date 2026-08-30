"""Point-spread-function kernels for the physics-consistent reconstruction

term (paper Sec. IV.G, Eq. 6-7).


The original renderer splats an isotropic Gaussian. Sec. II.B of the paper

measures the actual point-spread function of this imaging system and finds two

departures from that model:


    * heavy tails. Four pixels from the center the measured profile retains

      12% of the peak where the fitted Gaussian predicts 1.3%; seven pixels out

      it retains 1.6% where the Gaussian predicts 2e-6.


    * side-to-side asymmetry. The two sides of the profile differ by a factor

      of two at four pixels and by a factor of eight at five pixels.


The kernels below restore those two degrees of freedom. All of them are

differentiable with respect to the offset vector, so the reconstruction term

keeps training the coordinate head exactly as before.


Fitting the measured shoulder. With the fitted core sigma = 1.36 px, an offset

of 4 px gives q = (4/1.36)^2 = 8.65. The Gaussian limit exp(-q/2) = 1.3%

reproduces the number quoted in Sec. II.B. The Moffat form (1 + q/(2*beta))^-beta

gives 10.0% at beta = 2.0 and 13.1% at beta = 1.5, so beta near 1.6 reproduces

the measured 12% shoulder. beta is learnable and `beta_init` only sets the

starting point.

"""


import math


import torch

import torch.nn as nn

import torch.nn.functional as F


class IsotropicGaussianKernel(nn.Module):

    """The original renderer profile, kept so that `kernel='isotropic'`

    reproduces previously trained runs bit for bit.


    The width lives on the parent `PSFForwardModel` as `log_sigma`, so the

    state_dict of an isotropic model is unchanged and old checkpoints load

    without remapping. This module therefore holds no parameters of its own.

    """


    analytic_norm = True


    def forward(self, delta, sigma=None, coords=None, image_shape=None):

        dist2 = delta.pow(2).sum(dim=-1)

        return torch.exp(-dist2 / (2 * sigma**2)) / (2 * math.pi * sigma**2)


class AnisotropicMoffatKernel(nn.Module):

    """Elliptical Moffat profile with a directional skew factor.


    The quadratic form q = d^T M d uses a Cholesky factor M = L L^T, so M stays

    positive definite for any parameter value:


        L = [[exp(a),  0     ],

             [b,       exp(c)]]


    The radial profile is Moffat rather than Gaussian,


        psi_core = (1 + q / (2 * beta)) ** (-beta),


    which tends to the Gaussian exp(-q/2) as beta grows and is heavy tailed for

    small beta. The skew factor exp(w . d) breaks the reflection symmetry of the

    profile about its center; w is bounded through a tanh so the factor cannot

    run away during training.

    """


    analytic_norm = False


    def __init__(self, sigma_init=1.5, beta_init=2.0, skew_scale=0.35):

        super().__init__()

        inv_sigma = 1.0 / max(float(sigma_init), 1e-3)

        # a and c are log scales of the two Cholesky diagonal entries, so the

        # initial M is (1/sigma^2) * I and the kernel starts isotropic.

        self.raw_chol = nn.Parameter(

            torch.tensor([math.log(inv_sigma), 0.0, math.log(inv_sigma)])

        )

        # softplus(raw_beta) + 1 keeps beta > 1, where the Moffat integral

        # converges in two dimensions.

        beta_offset = max(float(beta_init) - 1.0, 1e-3)

        self.raw_beta = nn.Parameter(

            torch.tensor(math.log(math.expm1(beta_offset)))

        )

        self.raw_skew = nn.Parameter(torch.zeros(2))

        self.skew_scale = float(skew_scale)


    @property

    def beta(self):

        return F.softplus(self.raw_beta) + 1.0


    @property

    def skew(self):

        return self.skew_scale * torch.tanh(self.raw_skew)


    def covariance_scales(self):

        """Returns the two principal widths in pixels, for logging."""

        a, b, c = self.raw_chol[0], self.raw_chol[1], self.raw_chol[2]

        lower = torch.stack(

            [

                torch.stack([a.exp(), torch.zeros_like(b)]),

                torch.stack([b, c.exp()]),

            ]

        )

        precision = lower @ lower.T

        eigvals = torch.linalg.eigvalsh(precision).clamp(min=1e-8)

        return eigvals.rsqrt()


    def forward(self, delta, sigma=None, coords=None, image_shape=None):

        a, b, c = self.raw_chol[0], self.raw_chol[1], self.raw_chol[2]

        dx = delta[..., 0]

        dy = delta[..., 1]

        # u = L^T d, so that q = |u|^2 = d^T L L^T d.

        u1 = a.exp() * dx + b * dy

        u2 = c.exp() * dy

        q = u1.pow(2) + u2.pow(2)


        beta = self.beta

        core = (1.0 + q / (2.0 * beta)).pow(-beta)


        skew = self.skew

        tilt = (skew[0] * dx + skew[1] * dy).clamp(-8.0, 8.0)

        return core * torch.exp(tilt)


class EmpiricalGridKernel(nn.Module):

    """Free-form kernel sampled on the reconstruction support.


    The profile is stored as one learnable value per support pixel and read at

    sub-pixel offsets by bilinear interpolation, so it can represent heavy

    tails, ellipticity and skew at once without committing to a parametric

    family. softplus keeps the profile non-negative.


    `init_profile` accepts a measured (2R+1, 2R+1) stacked profile; without one

    the grid starts from an isotropic Gaussian of width `sigma_init`, so early

    training behaves like the original renderer.

    """


    analytic_norm = False


    def __init__(self, recon_radius=6, sigma_init=1.5, init_profile=None):

        super().__init__()

        self.recon_radius = int(recon_radius)

        size = 2 * self.recon_radius + 1


        if init_profile is None:

            coords = torch.arange(size, dtype=torch.float32) - self.recon_radius

            gy, gx = torch.meshgrid(coords, coords, indexing="ij")

            profile = torch.exp(-(gx.pow(2) + gy.pow(2)) / (2 * sigma_init**2))

        else:

            profile = torch.as_tensor(init_profile, dtype=torch.float32)

            if profile.shape != (size, size):

                raise ValueError(

                    f"init_profile must have shape {(size, size)}, got {tuple(profile.shape)}"

                )

            profile = profile.clamp(min=0.0)


        peak = profile.max().clamp(min=1e-8)

        profile = (profile / peak).clamp(min=1e-6)

        # Invert softplus so that softplus(raw_grid) reproduces the profile.

        self.raw_grid = nn.Parameter(torch.log(torch.expm1(profile)))


    @property

    def profile(self):

        return F.softplus(self.raw_grid)


    def forward(self, delta, sigma=None, coords=None, image_shape=None):

        radius = float(self.recon_radius)

        # grid_sample wants normalized coordinates in [-1, 1] over the support.

        gx = (delta[..., 0] / radius).clamp(-1.0, 1.0)

        gy = (delta[..., 1] / radius).clamp(-1.0, 1.0)


        shape = gx.shape

        grid = torch.stack([gx.reshape(-1), gy.reshape(-1)], dim=-1)

        grid = grid.view(1, -1, 1, 2)


        source = self.profile.view(1, 1, *self.profile.shape).to(

            dtype=grid.dtype, device=grid.device

        )

        sampled = F.grid_sample(

            source, grid, mode="bilinear", padding_mode="zeros", align_corners=True

        )

        return sampled.view(shape)


class FieldDependentKernel(nn.Module):

    """Elliptical Moffat whose shape varies across the register.


    Why this exists. The three kernels above all render psi(x - c): one profile

    shared by every site. That is wrong for this instrument, and the paper says

    so itself in the introduction, "central sites appear relatively compact and

    approximately circular, whereas peripheral sites often become elongated or

    comet-like". Two physical effects produce exactly that pattern and both

    scale with distance from a center:


      * optical aberrations. Coma grows linearly with field radius and points

        radially, producing a one-sided flare -- the "comet". Astigmatism grows

        quadratically and elongates the profile along the radial or tangential

        direction depending on which side of best focus the field point sits.

        Spherical aberration and defocus are field independent and stay in the

        base profile.


      * RF micromotion. Its amplitude grows with distance from the RF null and

        its direction is set by the RF field, so it smears peripheral ions along

        one fixed axis. The 28 DC segments of this trap null the micromotion

        perpendicular to the crystal plane, which leaves the in-plane component

        to grow toward the edge of the register.


    Why a polynomial in the site coordinate rather than a free per-site kernel.

    A per-site kernel would need 300 times the parameters of a single kernel and

    would have no way to share statistics between neighbouring sites. Aberration

    theory already tells us the radial dependence of each term (linear for coma,

    quadratic for astigmatism) and the orientation (radial for both), so only

    the amplitudes are unknown. That reduces the whole field dependence to seven

    numbers, and every one of them is a quantity the optics or the trap model can

    be checked against independently.


    Parameter budget beyond the base profile:


        raw_center          2   field center c0, as an offset from the frame

                                center in pixels

        raw_coma            1   coma amplitude; sign flips the flare between

                                pointing away from and toward c0

        raw_astig           2   astigmatism, as separate non-negative radial and

                                tangential variances (see below)

        raw_micromotion     2   in-plane micromotion amplitude and axis angle


    On the astigmatism count. Describing astigmatism with a single signed number

    cannot be done while keeping the covariance positive definite for both

    orientations, because a negative coefficient would subtract variance. Real

    astigmatism separates the radial and tangential focal planes, so at any fixed

    focus both directions are broader than best focus but by different amounts.

    Two non-negative coefficients express that directly and stay positive

    definite by construction.


    Why the terms compose in covariance space. Each broadening mechanism adds an

    independent displacement to the photon landing position, so their covariances

    add: Sigma(c) = Sigma_0 + var_radial * u u^T + var_tangential * v v^T +

    var_micromotion * m m^T. The quadratic form is then q = d^T Sigma(c)^-1 d.

    Adding the terms to the precision matrix instead would be wrong in the

    opposite direction: a larger q makes the profile fall off faster, so an

    aberration that is supposed to broaden the image would narrow it.

    """


    analytic_norm = False


    def __init__(

        self,

        sigma_init=1.5,

        beta_init=2.0,

        coma_scale=0.6,

        detach_field_coords=True,

    ):

        super().__init__()

        # Field-independent base, carrying defocus, spherical aberration and the
        # pixel box. Unlike AnisotropicMoffatKernel this Cholesky factor
        # parameterizes the COVARIANCE, not the precision, so that the
        # field-dependent variances below can simply be added to it.

        sigma = max(float(sigma_init), 1e-3)

        self.raw_chol = nn.Parameter(

            torch.tensor([math.log(sigma), 0.0, math.log(sigma)])

        )

        beta_offset = max(float(beta_init) - 1.0, 1e-3)

        self.raw_beta = nn.Parameter(

            torch.tensor(math.log(math.expm1(beta_offset)))

        )


        # Field-dependent terms. Coma starts exactly at zero because tanh(0) = 0.
        # The two variance amplitudes cannot start at exactly zero and stay
        # trainable: softplus is non-negative, which is what keeps the covariance
        # positive definite, but its gradient dies as its output approaches zero.
        # They start at softplus(-3) = 0.049 px^2 instead, which is 2.6% of the
        # initial sigma^2 = 1.85 px^2 and therefore negligible against the base
        # profile, while sigmoid(-3) = 0.047 leaves a workable gradient.

        self.raw_center = nn.Parameter(torch.zeros(2))

        self.raw_coma = nn.Parameter(torch.zeros(()))

        self.raw_astig = nn.Parameter(torch.full((2,), -3.0))

        self.raw_micromotion = nn.Parameter(torch.tensor([-3.0, 0.0]))


        self.coma_scale = float(coma_scale)

        self.detach_field_coords = bool(detach_field_coords)


    @property

    def beta(self):

        return F.softplus(self.raw_beta) + 1.0


    def _field_geometry(self, coords, image_shape):

        """Returns the normalized field radius and the radial unit vector.


        The radius is normalized by half the frame diagonal so it lands in

        roughly [0, 1] over the register. That only fixes the units of the

        amplitude coefficients; it keeps them O(1) and comparable to each other

        instead of differing by the powers of a pixel count.

        """

        H, W = image_shape

        center = torch.tensor(

            [(W - 1) / 2.0, (H - 1) / 2.0], device=coords.device, dtype=coords.dtype

        )

        c0 = center + self.raw_center

        offset = coords - c0

        scale = 0.5 * math.sqrt((W - 1) ** 2 + (H - 1) ** 2)

        radius = offset.norm(dim=-1, keepdim=True) / max(scale, 1.0)

        # The unit vector is undefined at the field center; the eps keeps the
        # gradient finite there and the terms it multiplies vanish anyway
        # because they all carry a factor of radius.

        radial = offset / offset.norm(dim=-1, keepdim=True).clamp(min=1e-6)

        return radius, radial


    def forward(self, delta, sigma=None, coords=None, image_shape=None):

        if coords is None or image_shape is None:

            raise ValueError(

                "FieldDependentKernel needs the site coordinates and the frame "
                "shape; PSFForwardModel passes both."

            )


        # The profile shape is a property of the optics at a field point, not
        # something the coordinate head should be free to move a site in order
        # to change. Detaching here keeps the gradient into pred_coords coming
        # from the geometric term (delta) alone and removes a spurious path
        # where a site drifts outward simply because a wider kernel there fits
        # the frame better.

        field_coords = coords.detach() if self.detach_field_coords else coords

        radius, radial = self._field_geometry(field_coords, image_shape)


        # (B,K,1) and (B,K,1,2) so they broadcast against delta (B,K,S,2).

        radius = radius.unsqueeze(-2)

        radial = radial.unsqueeze(-2)

        tangential = torch.stack([-radial[..., 1], radial[..., 0]], dim=-1)


        dx = delta[..., 0]

        dy = delta[..., 1]


        # Base covariance Sigma_0 = L L^T, positive definite for any parameter
        # value because the diagonal of L is exponentiated.

        a, b, c = self.raw_chol[0], self.raw_chol[1], self.raw_chol[2]

        s00 = (2 * a).exp()

        s01 = a.exp() * b

        s11 = b.pow(2) + (2 * c).exp()


        # Astigmatism separates the radial and tangential focal planes, so it
        # adds variance along both axes, each growing as radius^2 with its own
        # non-negative amplitude.

        astig = F.softplus(self.raw_astig)

        var_r = (astig[0] * radius.pow(2)).squeeze(-1)

        var_t = (astig[1] * radius.pow(2)).squeeze(-1)


        # Micromotion smears the ion along a fixed axis with an amplitude that
        # grows with distance from the RF null. The exposure-averaged
        # distribution along that axis is an arcsine profile; this term keeps
        # its second moment (variance = A^2 / 2 for half-amplitude A) but not
        # its double-horned shape, which no quadratic form can represent. The RF
        # null is assumed to coincide with the field center c0; if the optics
        # and the trap are not concentric, the amplitude absorbs the mismatch.

        amp = F.softplus(self.raw_micromotion[0])

        angle = self.raw_micromotion[1]

        mx, my = torch.cos(angle), torch.sin(angle)

        var_m = 0.5 * (amp * radius.squeeze(-1)).pow(2)


        # Independent displacements have additive covariances.

        ux, uy = radial[..., 0], radial[..., 1]

        vx, vy = tangential[..., 0], tangential[..., 1]

        cov00 = s00 + var_r * ux.pow(2) + var_t * vx.pow(2) + var_m * mx.pow(2)

        cov01 = s01 + var_r * ux * uy + var_t * vx * vy + var_m * mx * my

        cov11 = s11 + var_r * uy.pow(2) + var_t * vy.pow(2) + var_m * my.pow(2)


        # q = d^T Sigma^-1 d, using the closed-form 2x2 inverse.

        det = (cov00 * cov11 - cov01.pow(2)).clamp(min=1e-8)

        q = (

            cov11 * dx.pow(2) - 2 * cov01 * dx * dy + cov00 * dy.pow(2)

        ) / det


        proj_r = (delta * radial).sum(dim=-1)


        beta = self.beta

        core = (1.0 + q / (2.0 * beta)).pow(-beta)


        # Coma: a one-sided flare along the radial direction whose strength
        # grows linearly with field radius, so it vanishes on axis exactly as
        # aberration theory requires. tanh bounds the coefficient and the clamp
        # keeps the exponential from overflowing on the far edge of the support.

        coma = self.coma_scale * torch.tanh(self.raw_coma)

        tilt = (coma * radius.squeeze(-1) * proj_r).clamp(-8.0, 8.0)


        return core * torch.exp(tilt)


def build_psf_kernel(kernel, sigma_init=1.5, recon_radius=6, beta_init=2.0, init_profile=None):

    """Factory used by `PSFForwardModel`.


    kernel: 'isotropic' | 'anisotropic' | 'empirical'.

    """


    if kernel == "isotropic":

        return IsotropicGaussianKernel()

    if kernel == "anisotropic":

        return AnisotropicMoffatKernel(sigma_init=sigma_init, beta_init=beta_init)

    if kernel == "empirical":

        return EmpiricalGridKernel(

            recon_radius=recon_radius, sigma_init=sigma_init, init_profile=init_profile

        )

    if kernel == "field_dependent":

        return FieldDependentKernel(sigma_init=sigma_init, beta_init=beta_init)

    raise ValueError(

        f"unknown PSF kernel '{kernel}'; expected 'isotropic', 'anisotropic', "

        f"'empirical' or 'field_dependent'"

    )
