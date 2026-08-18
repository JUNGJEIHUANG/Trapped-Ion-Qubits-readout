


import torch

import torch.nn as nn

import torch.nn.functional as F


class TransformerBlock(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4.0, dropout=0.0):

        super().__init__()

        self.norm1 = nn.LayerNorm(dim)

        self.attn  = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)

        self.norm2 = nn.LayerNorm(dim)

        hidden = int(dim * mlp_ratio)

        self.mlp = nn.Sequential(

            nn.Linear(dim, hidden), nn.GELU(), nn.Dropout(dropout),

            nn.Linear(hidden, dim), nn.Dropout(dropout),

        )


    def forward(self, x):

        h = self.norm1(x)

        h, _ = self.attn(h, h, h, need_weights=False)

        x = x + h

        x = x + self.mlp(self.norm2(x))

        return x


def _sincos2d(h, w, dim, device):

    assert dim % 4 == 0

    half = dim // 2

    freq = 1.0 / (10000.0 ** (2.0 * torch.arange(half // 2, dtype=torch.float32, device=device) / half))

    ey = torch.outer(torch.arange(h, dtype=torch.float32, device=device), freq)

    ex = torch.outer(torch.arange(w, dtype=torch.float32, device=device), freq)

    py = torch.cat([ey.sin(), ey.cos()], dim=-1).unsqueeze(1).expand(h, w, half)

    px = torch.cat([ex.sin(), ex.cos()], dim=-1).unsqueeze(0).expand(h, w, half)

    return torch.cat([py, px], dim=-1).reshape(1, h * w, dim)


class PatchEmbed(nn.Module):

    def __init__(self, in_channels, embed_dim, patch_size):

        super().__init__()

        self.patch_size = patch_size

        self.proj = nn.Conv2d(in_channels, embed_dim, patch_size, stride=patch_size)


    def forward(self, x):

        x = self.proj(x)

        B, D, h, w = x.shape

        return x.flatten(2).transpose(1, 2), h, w


def _pup_decoder(embed_dim, out_channels):


    c = embed_dim

    return nn.Sequential(

        nn.Conv2d(c, c // 2, 3, padding=1), nn.ReLU(inplace=True),

        nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),

        nn.Conv2d(c // 2, c // 2, 3, padding=1), nn.ReLU(inplace=True),

        nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),

        nn.Conv2d(c // 2, c // 4, 3, padding=1), nn.ReLU(inplace=True),

        nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),

        nn.Conv2d(c // 4, c // 4, 3, padding=1), nn.ReLU(inplace=True),

        nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),

        nn.Conv2d(c // 4, out_channels, 1),

    )


class SETR(nn.Module):

    def __init__(

        self,

        in_channels:  int   = 1,

        out_channels: int   = 1,

        patch_size:   int   = 16,

        embed_dim:    int   = 256,

        depth:        int   = 12,

        num_heads:    int   = 8,

        mlp_ratio:    float = 4.0,

        dropout:      float = 0.0,

        decoder:      str   = "pup",

    ):

        super().__init__()

        self.patch_size  = patch_size

        self.embed_dim   = embed_dim

        self.decoder_type = decoder


        self.patch_embed = PatchEmbed(in_channels, embed_dim, patch_size)

        self.blocks = nn.ModuleList(

            [TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout) for _ in range(depth)]

        )

        self.norm = nn.LayerNorm(embed_dim)


        if decoder == "pup":

            self.decoder = _pup_decoder(embed_dim, out_channels)

        else:

            self.decoder = nn.Conv2d(embed_dim, out_channels, 1)


        self._pos_cache: dict = {}


    def _pos(self, h, w, device):

        key = (h, w)

        if key not in self._pos_cache:

            self._pos_cache[key] = _sincos2d(h, w, self.embed_dim, device)

        return self._pos_cache[key]


    def forward(self, x):

        B, _, H, W = x.shape

        tokens, h, w = self.patch_embed(x)

        tokens = tokens + self._pos(h, w, x.device)


        for blk in self.blocks:

            tokens = blk(tokens)

        tokens = self.norm(tokens)


        feat = tokens.transpose(1, 2).reshape(B, self.embed_dim, h, w)


        if self.decoder_type == "pup":

            out = self.decoder(feat)

        else:

            out = self.decoder(feat)


        if out.shape[2:] != (H, W):

            out = F.interpolate(out, size=(H, W), mode="bilinear", align_corners=False)

        return out

