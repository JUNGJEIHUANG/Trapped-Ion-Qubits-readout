import torch

import torch.nn as nn

import torch.nn.functional as F


class AdaptiveTop16Tokenizer(nn.Module):

    """Offset-corrected adaptive Top-16 tokenization (Site-AIT paper Sec. IV.C, App. J).

    Converts a fused feature field F_i in R^{C x H x W} into one 256-dim image
    token per calibrated site: a learned per-site offset recenters a 13x13
    candidate window on the frame-specific ion position, a shared scorer ranks
    the 169 candidates, a Gumbel-perturbed hard Top-16 gate with a
    straight-through softmax surrogate selects 16 of them, and a learnable
    PSF-inspired mask reweights only the selected features before concatenation.
    """

    def __init__(

        self,

        in_channels=16,

        radius=6,

        num_select=16,

        selector_hidden=32,

        offset_hidden=32,

        offset_max_px=3.0,

        tau_start=0.8,

        tau_end=0.1,

        disable_offset=False,

        disable_psf_mask=False,

    ):

        super().__init__()

        self.in_channels = in_channels

        self.radius = radius

        self.window = 2 * radius + 1

        self.num_candidates = self.window * self.window

        self.num_select = num_select

        self.offset_max_px = float(offset_max_px)

        self.tau_start = float(tau_start)

        self.tau_end = float(tau_end)

        self.disable_offset = bool(disable_offset)

        self.disable_psf_mask = bool(disable_psf_mask)


        self.g_delta = nn.Sequential(

            nn.Linear(in_channels, offset_hidden),

            nn.ReLU(inplace=True),

            nn.Linear(offset_hidden, 2),

        )

        nn.init.zeros_(self.g_delta[-1].weight)

        nn.init.zeros_(self.g_delta[-1].bias)


        self.g_sel = nn.Sequential(

            nn.Linear(in_channels, selector_hidden),

            nn.ReLU(inplace=True),

            nn.Linear(selector_hidden, 1),

        )


        self.psf_mask = nn.Parameter(torch.zeros(self.window, self.window))


        offsets = torch.stack(

            torch.meshgrid(

                torch.arange(-radius, radius + 1, dtype=torch.float32),

                torch.arange(-radius, radius + 1, dtype=torch.float32),

                indexing="ij",

            ),

            dim=-1,

        ).reshape(-1, 2)

        self.register_buffer("_local_offsets_rc", offsets, persistent=False)


        self._tau = self.tau_start


    def set_temperature_progress(self, progress):

        """progress in [0,1]; the trainer calls this once per epoch to linearly

        anneal the softmax surrogate temperature from tau_start to tau_end."""

        progress = min(max(float(progress), 0.0), 1.0)

        self._tau = self.tau_start + (self.tau_end - self.tau_start) * progress


    @property

    def output_dim(self):

        return self.num_select * self.in_channels


    def forward(self, feature_map, site_coords):

        """

        feature_map: (B, C, H, W) fused feature field, C == in_channels.

        site_coords: (B, K, 2) nominal site coordinates stored as (x=col, y=row)

            pixels, matching this repo's existing coordinate convention (the

            paper's (u,v) row/col notation is purely a math-indexing choice and

            is equivalent under the axis-aligned square candidate window used

            here).

        Returns (tokens, aux): tokens is (B, K, num_select * C); aux carries the

        predicted tokenization offsets and raw selector logits for regularization.

        """

        B, C, H, W = feature_map.shape

        if C != self.in_channels:

            raise ValueError(f"Expected {self.in_channels} feature channels, got {C}")

        K = site_coords.shape[1]

        device = feature_map.device

        dtype = feature_map.dtype


        c0_xy = site_coords


        nearest_x = c0_xy[..., 0].round().clamp(0, W - 1).long()

        nearest_y = c0_xy[..., 1].round().clamp(0, H - 1).long()

        batch_idx = torch.arange(B, device=device).view(B, 1).expand(B, K)

        feat_at_nominal = feature_map[batch_idx, :, nearest_y, nearest_x]


        if self.disable_offset:

            tok_offset = torch.zeros(B, K, 2, device=device, dtype=dtype)

        else:

            tok_offset = self.offset_max_px * torch.tanh(self.g_delta(feat_at_nominal))


        corrected_center = c0_xy + tok_offset


        local_xy = torch.stack(

            [self._local_offsets_rc[:, 1], self._local_offsets_rc[:, 0]], dim=-1

        ).to(device=device, dtype=dtype)

        sample_xy = corrected_center.unsqueeze(2) + local_xy.view(1, 1, -1, 2)


        grid_x = sample_xy[..., 0] / max(W - 1, 1) * 2.0 - 1.0

        grid_y = sample_xy[..., 1] / max(H - 1, 1) * 2.0 - 1.0

        grid = torch.stack([grid_x, grid_y], dim=-1)

        flat_grid = grid.view(B, K * self.num_candidates, 1, 2)


        sampled = F.grid_sample(

            feature_map, flat_grid, mode="bilinear", padding_mode="zeros", align_corners=True

        )

        sampled = sampled.squeeze(-1).transpose(1, 2)

        P = sampled.view(B, K, self.num_candidates, C)


        logits = self.g_sel(P).squeeze(-1)


        if self.training:

            uniform = torch.rand_like(logits).clamp_(1e-20, 1.0 - 1e-20)

            gumbel = -torch.log(-torch.log(uniform))

            noisy_logits = logits + gumbel

        else:

            noisy_logits = logits


        top_val, top_idx = torch.topk(noisy_logits, self.num_select, dim=-1)

        hard = torch.zeros_like(noisy_logits).scatter_(-1, top_idx, 1.0)


        if self.training:

            soft = torch.softmax(noisy_logits / self._tau, dim=-1)

            pi = hard - soft.detach() + soft

        else:

            pi = hard


        if self.disable_psf_mask:

            omega = torch.full(

                (self.num_candidates,), 1.0 / self.num_select, device=device, dtype=dtype

            )

        else:

            omega = torch.softmax(self.psf_mask.view(-1), dim=0).to(dtype=dtype)

        omega_full = omega.view(1, 1, -1).expand(B, K, -1)


        order = torch.argsort(top_val, dim=-1, descending=True)

        ordered_idx = torch.gather(top_idx, -1, order)


        pi_sel = torch.gather(pi, -1, ordered_idx)

        omega_sel = torch.gather(omega_full, -1, ordered_idx)

        gate = pi_sel * omega_sel


        feat_sel = torch.gather(

            P, 2, ordered_idx.unsqueeze(-1).expand(-1, -1, -1, C)

        )


        q = gate.unsqueeze(-1) * feat_sel

        token = q.reshape(B, K, self.num_select * C)


        aux = {

            "tok_offset": tok_offset,

            "sel_logits": logits,

        }

        return token, aux


class GlobalCrossAttnTokenizer(nn.Module):

    """'w/ global cross-attn' ablation (Table VI): replaces adaptive Top-16

    selection and post-selection PSF weighting with a single generic global

    cross-attention layer. Each site's coordinate-embedded query attends over

    every spatial position of the fused feature field (no local windowing, no

    top-k gating). The paper only states that this row substitutes "generic

    global cross-attention" for the proposed tokenization; the exact mechanism

    is an engineering interpretation, consistent with the paper's reported

    higher latency for this row (11.80ms vs. 4.50ms) since attention here spans

    the full H*W feature map rather than a 13x13 window.

    """

    def __init__(self, in_channels=16, out_dim=256, num_heads=8):

        super().__init__()

        self.in_channels = in_channels

        self.out_dim = out_dim

        self.query_coord = nn.Sequential(

            nn.Linear(2, out_dim),

            nn.ReLU(inplace=True),

            nn.Linear(out_dim, out_dim),

        )

        self.kv_proj = nn.Linear(in_channels, out_dim)

        self.attn = nn.MultiheadAttention(out_dim, num_heads, batch_first=True)


    def set_temperature_progress(self, progress):

        return


    @property

    def output_dim(self):

        return self.out_dim


    def forward(self, feature_map, site_coords):

        B, C, H, W = feature_map.shape

        if C != self.in_channels:

            raise ValueError(f"Expected {self.in_channels} feature channels, got {C}")

        K = site_coords.shape[1]


        coord01 = torch.stack(

            [

                site_coords[..., 0] / max(W - 1, 1),

                site_coords[..., 1] / max(H - 1, 1),

            ],

            dim=-1,

        ).clamp(0, 1)

        query = self.query_coord(coord01)


        kv = feature_map.flatten(2).transpose(1, 2)

        kv = self.kv_proj(kv)


        token, _ = self.attn(query, kv, kv, need_weights=False)


        aux = {

            "tok_offset": torch.zeros(

                B, K, 2, device=feature_map.device, dtype=feature_map.dtype

            ),

        }

        return token, aux

