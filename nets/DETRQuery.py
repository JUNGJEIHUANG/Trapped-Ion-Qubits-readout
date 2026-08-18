import torch

import torch.nn as nn

import torch.nn.functional as F

from nets.dwunet_backbone import DWUNetBackboneDecoder


class DETRStyleQuery(DWUNetBackboneDecoder):

    """Coordinate-aware baseline (Site-AIT paper App. F, 'DETR-style Query').

    One learnable query per nominal site, the same DW-UNet backbone as

    Site-AIT, and cross-attention over a generic learned 13x13 window in place

    of Site-AIT's PSF-guided adaptive Top-16 selection and post-selection

    weighting, followed by a Transformer encoder mixing queries and a linear

    head for the bright-state logit. This baseline does not refine site

    coordinates (App. F only describes a classification head), so pred_coords

    is passed through unchanged.

    Selected hyperparameters (App. F): eta=1e-4, D=256, L=4, H=8, 13x13

    template. Full search-space is documented in REPRODUCIBILITY.md.

    """


    def __init__(

        self,

        num_ions=300,

        token_dim=256,

        num_heads=8,

        num_layers=4,

        window_radius=6,

        in_channels=17,

        fused_channels=16,

        fuse_hidden=64,

    ):

        super().__init__()

        self.num_ions = num_ions

        self.window_radius = window_radius

        self.window = 2 * window_radius + 1

        self.num_candidates = self.window * self.window


        self.fuse = nn.Sequential(

            nn.Conv2d(in_channels, fuse_hidden, kernel_size=3, padding=1, bias=False),

            nn.BatchNorm2d(fuse_hidden),

            nn.ReLU(inplace=True),

            nn.Conv2d(fuse_hidden, fused_channels, kernel_size=1),

        )


        self.query_embed = nn.Embedding(num_ions, token_dim)

        self.coord_embed = nn.Sequential(

            nn.Linear(2, token_dim),

            nn.ReLU(inplace=True),

            nn.Linear(token_dim, token_dim),

        )

        self.kv_proj = nn.Linear(fused_channels, token_dim)

        self.cross_attn = nn.MultiheadAttention(token_dim, num_heads, batch_first=True)


        offsets = torch.stack(

            torch.meshgrid(

                torch.arange(-window_radius, window_radius + 1, dtype=torch.float32),

                torch.arange(-window_radius, window_radius + 1, dtype=torch.float32),

                indexing="ij",

            ),

            dim=-1,

        ).reshape(-1, 2)

        self.register_buffer("_local_offsets_rc", offsets, persistent=False)


        encoder_layer = nn.TransformerEncoderLayer(

            d_model=token_dim,

            nhead=num_heads,

            dim_feedforward=token_dim * 2,

            dropout=0.0,

            batch_first=True,

            norm_first=True,

        )

        self.query_interaction = nn.TransformerEncoder(

            encoder_layer, num_layers=num_layers, enable_nested_tensor=False

        )


        self.state_head = nn.Linear(token_dim, 1)


        self._init_weights()


    def forward(self, x, site_coords=None, return_dict=False):

        if site_coords is None:

            raise ValueError("DETRStyleQuery requires site_coords")


        raw_input, up4, mask_logits = self.extract_features(x)

        B, _, H, W = raw_input.shape

        if site_coords.dim() == 2:

            site_coords = site_coords.unsqueeze(0).expand(B, -1, -1)

        site_coords = site_coords.to(device=raw_input.device, dtype=raw_input.dtype)

        K = site_coords.shape[1]


        if up4.shape[-2:] != (H, W):

            up4 = F.interpolate(up4, size=(H, W), mode="bilinear", align_corners=True)

        fused = self.fuse(torch.cat([up4, raw_input], dim=1))


        local_xy = torch.stack(

            [self._local_offsets_rc[:, 1], self._local_offsets_rc[:, 0]], dim=-1

        ).to(device=raw_input.device, dtype=raw_input.dtype)

        sample_xy = site_coords.unsqueeze(2) + local_xy.view(1, 1, -1, 2)


        grid_x = sample_xy[..., 0] / max(W - 1, 1) * 2.0 - 1.0

        grid_y = sample_xy[..., 1] / max(H - 1, 1) * 2.0 - 1.0

        grid = torch.stack([grid_x, grid_y], dim=-1)

        flat_grid = grid.view(B, K * self.num_candidates, 1, 2)


        sampled = F.grid_sample(

            fused, flat_grid, mode="bilinear", padding_mode="zeros", align_corners=True

        )

        sampled = sampled.squeeze(-1).transpose(1, 2).view(B, K, self.num_candidates, -1)

        kv = self.kv_proj(sampled)


        ids = torch.arange(K, device=raw_input.device)

        coord01 = torch.stack(

            [

                site_coords[..., 0] / max(W - 1, 1),

                site_coords[..., 1] / max(H - 1, 1),

            ],

            dim=-1,

        ).clamp(0, 1)

        query = self.query_embed(ids).unsqueeze(0).expand(B, -1, -1) + self.coord_embed(coord01)


        query_flat = query.reshape(B * K, 1, -1)

        kv_flat = kv.reshape(B * K, self.num_candidates, -1)

        attended, _ = self.cross_attn(query_flat, kv_flat, kv_flat, need_weights=False)

        token = attended.reshape(B, K, -1) + query


        token = self.query_interaction(token)


        bright_logit = self.state_head(token).squeeze(-1)


        return {

            "tokens": token,

            "bright_logit": bright_logit,

            "pred_coords": site_coords,

            "mask_logits": mask_logits,

        }

