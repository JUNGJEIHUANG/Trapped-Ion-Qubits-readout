import torch

import torch.nn as nn

import torch.nn.functional as F

from nets.dwunet_backbone import DWUNetBackboneDecoder


class MAB(nn.Module):

    """Multihead Attention Block (Lee et al., 2019, 'Set Transformer')."""


    def __init__(self, dim_q, dim_k, dim_v, num_heads):

        super().__init__()

        self.fc_q = nn.Linear(dim_q, dim_v)

        self.fc_k = nn.Linear(dim_k, dim_v)

        self.fc_v = nn.Linear(dim_k, dim_v)

        self.attn = nn.MultiheadAttention(dim_v, num_heads, batch_first=True)

        self.ln0 = nn.LayerNorm(dim_v)

        self.fc_o = nn.Sequential(

            nn.Linear(dim_v, dim_v), nn.ReLU(inplace=True), nn.Linear(dim_v, dim_v)

        )

        self.ln1 = nn.LayerNorm(dim_v)


    def forward(self, q_in, kv_in):

        q = self.fc_q(q_in)

        k = self.fc_k(kv_in)

        v = self.fc_v(kv_in)

        attn_out, _ = self.attn(q, k, v, need_weights=False)

        o = self.ln0(q + attn_out)

        o = self.ln1(o + self.fc_o(o))

        return o


class ISAB(nn.Module):

    """Induced Set Attention Block (Lee et al., 2019): reduces the O(K^2) cost

    of full self-attention over K sites to O(K*M) via M learnable inducing

    points."""


    def __init__(self, dim, num_heads, num_inducing):

        super().__init__()

        self.inducing_points = nn.Parameter(torch.randn(1, num_inducing, dim) * 0.02)

        self.mab0 = MAB(dim, dim, dim, num_heads)

        self.mab1 = MAB(dim, dim, dim, num_heads)


    def forward(self, x):

        b = x.shape[0]

        induced = self.inducing_points.expand(b, -1, -1)

        h = self.mab0(induced, x)

        return self.mab1(x, h)


class SetTransformerReadout(DWUNetBackboneDecoder):

    """Coordinate-aware baseline (Site-AIT paper App. F, 'Set Transformer').

    Per-site features are bilinearly sampled from the DW-UNet feature map at

    the nominal coordinate only (no adaptive PSF sampling, no frame-specific

    offset correction), augmented with a coordinate embedding, and modeled

    jointly across sites with 2 ISAB blocks (Lee et al., 2019). This baseline

    does not refine site coordinates or use reconstruction consistency.

    Selected hyperparameters (App. F): eta=3e-4, D=256, 2 ISAB blocks, M=32

    inducing points, point-bilinear sampling.

    """


    def __init__(

        self,

        num_ions=300,

        token_dim=256,

        num_heads=8,

        num_isab=2,

        num_inducing=32,

        in_channels=17,

        fused_channels=16,

        fuse_hidden=64,

    ):

        super().__init__()

        self.num_ions = num_ions


        self.fuse = nn.Sequential(

            nn.Conv2d(in_channels, fuse_hidden, kernel_size=3, padding=1, bias=False),

            nn.BatchNorm2d(fuse_hidden),

            nn.ReLU(inplace=True),

            nn.Conv2d(fuse_hidden, fused_channels, kernel_size=1),

        )


        self.feature_proj = nn.Linear(fused_channels, token_dim)

        self.coord_embed = nn.Sequential(

            nn.Linear(2, token_dim),

            nn.ReLU(inplace=True),

            nn.Linear(token_dim, token_dim),

        )


        self.isabs = nn.ModuleList(

            [ISAB(token_dim, num_heads, num_inducing) for _ in range(num_isab)]

        )


        self.state_head = nn.Linear(token_dim, 1)


        self._init_weights()


    def forward(self, x, site_coords=None, return_dict=False):

        if site_coords is None:

            raise ValueError("SetTransformerReadout requires site_coords")


        raw_input, up4, mask_logits = self.extract_features(x)

        B, _, H, W = raw_input.shape

        if site_coords.dim() == 2:

            site_coords = site_coords.unsqueeze(0).expand(B, -1, -1)

        site_coords = site_coords.to(device=raw_input.device, dtype=raw_input.dtype)

        K = site_coords.shape[1]


        if up4.shape[-2:] != (H, W):

            up4 = F.interpolate(up4, size=(H, W), mode="bilinear", align_corners=True)

        fused = self.fuse(torch.cat([up4, raw_input], dim=1))


        grid_x = site_coords[..., 0] / max(W - 1, 1) * 2.0 - 1.0

        grid_y = site_coords[..., 1] / max(H - 1, 1) * 2.0 - 1.0

        grid = torch.stack([grid_x, grid_y], dim=-1).view(B, K, 1, 2)


        sampled = F.grid_sample(

            fused, grid, mode="bilinear", padding_mode="border", align_corners=True

        )

        sampled = sampled.squeeze(-1).transpose(1, 2)


        coord01 = torch.stack(

            [

                site_coords[..., 0] / max(W - 1, 1),

                site_coords[..., 1] / max(H - 1, 1),

            ],

            dim=-1,

        ).clamp(0, 1)

        token = self.feature_proj(sampled) + self.coord_embed(coord01)


        for isab in self.isabs:

            token = isab(token)


        bright_logit = self.state_head(token).squeeze(-1)


        return {

            "tokens": token,

            "bright_logit": bright_logit,

            "pred_coords": site_coords,

            "mask_logits": mask_logits,

        }

