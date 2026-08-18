import math

import torch

import torch.nn as nn

import torch.nn.functional as F


class DoubleConv(nn.Module):

    def __init__(self, in_ch, out_ch):

        super().__init__()

        self.net = nn.Sequential(

            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),

            nn.BatchNorm2d(out_ch),

            nn.ReLU(inplace=True),

            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),

            nn.BatchNorm2d(out_ch),

            nn.ReLU(inplace=True),

        )


    def forward(self, x):

        return self.net(x)


class TransformerBlock(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4.0, dropout=0.0):

        super().__init__()

        self.norm1 = nn.LayerNorm(dim)

        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)

        self.norm2 = nn.LayerNorm(dim)

        hidden = int(dim * mlp_ratio)

        self.mlp = nn.Sequential(

            nn.Linear(dim, hidden),

            nn.GELU(),

            nn.Dropout(dropout),

            nn.Linear(hidden, dim),

            nn.Dropout(dropout),

        )


    def forward(self, x):

        h = self.norm1(x)

        h, _ = self.attn(h, h, h, need_weights=False)

        x = x + h

        x = x + self.mlp(self.norm2(x))

        return x


def _sincos_pos_embed(h, w, dim):


    assert dim % 4 == 0

    half = dim // 2


    gy = torch.arange(h, dtype=torch.float32)

    gx = torch.arange(w, dtype=torch.float32)

    freq = torch.arange(half // 2, dtype=torch.float32)

    freq = 1.0 / (10000.0 ** (2.0 * freq / half))


    ey = torch.outer(gy, freq)

    ex = torch.outer(gx, freq)


    py = torch.cat([ey.sin(), ey.cos()], dim=-1)

    px = torch.cat([ex.sin(), ex.cos()], dim=-1)


    py = py.unsqueeze(1).expand(h, w, half)

    px = px.unsqueeze(0).expand(h, w, half)

    pos = torch.cat([py, px], dim=-1)

    return pos.reshape(1, h * w, dim)


class ViTUNet(nn.Module):


    def __init__(

        self,

        in_channels: int = 1,

        out_channels: int = 1,

        base_channels: int = 32,

        vit_dim: int = 256,

        vit_depth: int = 6,

        vit_heads: int = 8,

        mlp_ratio: float = 4.0,

        dropout: float = 0.0,

    ):

        super().__init__()

        c = base_channels


        self.enc1 = DoubleConv(in_channels, c)

        self.enc2 = DoubleConv(c,     c * 2)

        self.enc3 = DoubleConv(c * 2, c * 4)

        self.pool = nn.MaxPool2d(2)


        self.proj_in = nn.Conv2d(c * 4, vit_dim, 1)


        self.vit_blocks = nn.ModuleList(

            [TransformerBlock(vit_dim, vit_heads, mlp_ratio, dropout) for _ in range(vit_depth)]

        )

        self.norm = nn.LayerNorm(vit_dim)


        self.proj_out = nn.Conv2d(vit_dim, c * 4, 1)


        self.up3 = nn.ConvTranspose2d(c * 4, c * 2, 2, stride=2)

        self.dec3 = DoubleConv(c * 4 + c * 2, c * 2)

        self.up2 = nn.ConvTranspose2d(c * 2, c, 2, stride=2)

        self.dec2 = DoubleConv(c * 2 + c, c)

        self.up1 = nn.ConvTranspose2d(c, c, 2, stride=2)

        self.dec1 = DoubleConv(c * 2, c)


        self.out_conv = nn.Conv2d(c, out_channels, 1)


        self._pos_cache: dict = {}


    @staticmethod

    def _align(x, ref):

        if x.shape[2:] != ref.shape[2:]:

            x = F.interpolate(x, size=ref.shape[2:], mode="bilinear", align_corners=False)

        return x


    def _get_pos_embed(self, h, w, device):

        key = (h, w)

        if key not in self._pos_cache:

            self._pos_cache[key] = _sincos_pos_embed(h, w, self.proj_in.out_channels).to(device)

        return self._pos_cache[key]


    def forward(self, x):


        e1 = self.enc1(x)

        e2 = self.enc2(self.pool(e1))

        e3 = self.enc3(self.pool(e2))

        f  = self.pool(e3)


        f = self.proj_in(f)

        B, D, h, w = f.shape


        tokens = f.flatten(2).transpose(1, 2)

        tokens = tokens + self._get_pos_embed(h, w, f.device)


        for blk in self.vit_blocks:

            tokens = blk(tokens)

        tokens = self.norm(tokens)


        f = tokens.transpose(1, 2).reshape(B, D, h, w)

        f = self.proj_out(f)


        d3 = self._align(self.up3(f), e3)

        d3 = self.dec3(torch.cat([e3, d3], dim=1))


        d2 = self._align(self.up2(d3), e2)

        d2 = self.dec2(torch.cat([e2, d2], dim=1))


        d1 = self._align(self.up1(d2), e1)

        d1 = self.dec1(torch.cat([e1, d1], dim=1))


        return self.out_conv(d1)

