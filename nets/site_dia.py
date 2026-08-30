import torch

import torch.nn as nn

import torch.nn.functional as F

from nets.site_tokenizer import AdaptiveTop16Tokenizer, GlobalCrossAttnTokenizer


class SiteDIAHead(nn.Module):

    """Physics-informed site-token inference head (Site-AIT paper Sec. IV).

    Composition follows Fig. 2 of the paper:
      1. noise-aware feature fusion psi([F_up, I]) -> C_F channels (Sec. IV.B).
      2. offset-corrected adaptive Top-16 tokenization -> one 256-dim image
         token per calibrated site (Sec. IV.C), or the 'w/ global cross-attn'
         ablation tokenizer in its place.
      3. lattice-aware token = W_x @ image_token + coord_embed(c0) (Sec. IV.D).
      4. joint cross-site inference via a Transformer encoder (Sec. IV.E).
      5. per-site state / coordinate / existence readout heads (Sec. IV.F).
    """


    def __init__(

        self,

        num_ions=300,

        in_channels=17,

        fused_channels=16,

        fuse_hidden=64,

        token_dim=256,

        num_heads=8,

        num_ion_attn_layers=1,

        tok_radius=6,

        tok_num_select=16,

        tok_offset_max_px=3.0,

        tau_start=0.8,

        tau_end=0.1,

        max_coord_offset=3.0,

        use_global_cross_attn=False,

        disable_offset=False,

        disable_psf_mask=False,

        disable_coord_head=False,

    ):

        super().__init__()

        if token_dim % num_heads != 0:

            raise ValueError("token_dim must be divisible by num_heads")


        self.num_ions = num_ions

        self.token_dim = token_dim

        self.max_coord_offset = float(max_coord_offset)

        self.disable_coord_head = bool(disable_coord_head)

        self.use_global_cross_attn = bool(use_global_cross_attn)


        # Sec. IV.B: noise-aware feature fusion; output has C_F=fused_channels
        # channels, matching the paper's tilde-F in R^{C_F x H x W}, C_F=16.
        self.fuse = nn.Sequential(

            nn.Conv2d(in_channels, fuse_hidden, kernel_size=3, padding=1, bias=False),

            nn.BatchNorm2d(fuse_hidden),

            nn.ReLU(inplace=True),

            nn.Conv2d(fuse_hidden, fused_channels, kernel_size=1),

        )


        if self.use_global_cross_attn:

            self.tokenizer = GlobalCrossAttnTokenizer(

                in_channels=fused_channels,

                out_dim=token_dim,

                num_heads=num_heads,

            )

        else:

            self.tokenizer = AdaptiveTop16Tokenizer(

                in_channels=fused_channels,

                radius=tok_radius,

                num_select=tok_num_select,

                offset_max_px=tok_offset_max_px,

                tau_start=tau_start,

                tau_end=tau_end,

                disable_offset=disable_offset,

                disable_psf_mask=disable_psf_mask,

            )


        self.token_proj = nn.Linear(self.tokenizer.output_dim, token_dim)


        self.coord_embed = nn.Sequential(

            nn.Linear(2, token_dim),

            nn.ReLU(inplace=True),

            nn.Linear(token_dim, token_dim),

        )


        if num_ion_attn_layers > 0:

            self.ion_interaction = nn.TransformerEncoder(

                nn.TransformerEncoderLayer(

                    d_model=token_dim,

                    nhead=num_heads,

                    dim_feedforward=token_dim * 2,

                    dropout=0.0,

                    batch_first=True,

                    norm_first=True,

                ),

                num_layers=num_ion_attn_layers,

                enable_nested_tensor=False,

            )

        else:

            self.ion_interaction = None


        self.state_head = nn.Sequential(

            nn.Linear(token_dim, token_dim),

            nn.ReLU(inplace=True),

            nn.Linear(token_dim, 1),

        )


        if not self.disable_coord_head:

            self.coord_head = nn.Sequential(

                nn.Linear(token_dim, token_dim),

                nn.ReLU(inplace=True),

                nn.Linear(token_dim, 2),

            )

        else:

            self.coord_head = None


        self.exist_head = nn.Sequential(

            nn.Linear(token_dim, token_dim),

            nn.ReLU(inplace=True),

            nn.Linear(token_dim, 1),

        )


    def _coords_unit01(self, coords, height, width):

        x = coords[..., 0] / max(width - 1, 1)

        y = coords[..., 1] / max(height - 1, 1)

        return torch.stack([x, y], dim=-1).clamp(0, 1)


    def set_temperature_progress(self, progress):

        """Forwards the Gumbel/softmax-surrogate temperature schedule (Sec. IV.C:

        tau linearly annealed 0.8 -> 0.1) to the tokenizer, when it supports one."""

        set_fn = getattr(self.tokenizer, "set_temperature_progress", None)

        if set_fn is not None:

            set_fn(progress)


    def forward(self, raw_image, feature_map, site_coords):


        B, _, H, W = raw_image.shape

        if site_coords.dim() == 2:

            site_coords = site_coords.unsqueeze(0).expand(B, -1, -1)

        site_coords = site_coords.to(device=raw_image.device, dtype=raw_image.dtype)

        K = site_coords.shape[1]

        if K > self.num_ions:

            raise ValueError(f"Received {K} sites, but head was built for {self.num_ions}")


        if feature_map.shape[-2:] != (H, W):

            feature_map = F.interpolate(feature_map, size=(H, W), mode="bilinear", align_corners=True)

        fused = self.fuse(torch.cat([feature_map, raw_image], dim=1))


        image_token, tok_aux = self.tokenizer(fused, site_coords)

        image_token = self.token_proj(image_token)


        coord01 = self._coords_unit01(site_coords, H, W)

        token = image_token + self.coord_embed(coord01)


        if self.ion_interaction is not None:

            token = self.ion_interaction(token)


        bright_logit = self.state_head(token).squeeze(-1)


        if self.coord_head is not None:

            coord_delta = self.max_coord_offset * torch.tanh(self.coord_head(token))

            pred_coords = site_coords + coord_delta

        else:

            coord_delta = torch.zeros_like(site_coords)

            pred_coords = site_coords


        exist_logit = self.exist_head(token).squeeze(-1)


        return {

            "tokens": token,

            "bright_logit": bright_logit,

            "pred_coords": pred_coords,

            "coord_delta": coord_delta,

            "exist_logit": exist_logit,

            "dia_tok_offset": tok_aux.get("tok_offset"),

            "dia_sel_logits": tok_aux.get("sel_logits"),

        }

