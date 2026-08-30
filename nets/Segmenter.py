


import torch

import torch.nn as nn

import torch.nn.functional as F


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

        self.proj = nn.Conv2d(in_channels, embed_dim, patch_size, stride=patch_size)


    def forward(self, x):

        x = self.proj(x)

        B, D, h, w = x.shape

        return x.flatten(2).transpose(1, 2), h, w


class TransformerBlock(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4.0, dropout=0.0):

        super().__init__()

        self.norm1 = nn.LayerNorm(dim)

        self.attn  = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)

        self.norm2 = nn.LayerNorm(dim)

        hid = int(dim * mlp_ratio)

        self.mlp = nn.Sequential(

            nn.Linear(dim, hid), nn.GELU(), nn.Dropout(dropout),

            nn.Linear(hid, dim), nn.Dropout(dropout),

        )


    def forward(self, x):

        h = self.norm1(x)

        h, _ = self.attn(h, h, h, need_weights=False)

        x = x + h

        x = x + self.mlp(self.norm2(x))

        return x


class MaskDecoderLayer(nn.Module):


    def __init__(self, dim, num_heads, mlp_ratio=4.0, dropout=0.0):

        super().__init__()

        self.norm1  = nn.LayerNorm(dim)

        self.self_a = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)

        self.norm2  = nn.LayerNorm(dim)

        self.cross  = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)

        self.norm3  = nn.LayerNorm(dim)

        hid = int(dim * mlp_ratio)

        self.mlp = nn.Sequential(

            nn.Linear(dim, hid), nn.GELU(), nn.Dropout(dropout),

            nn.Linear(hid, dim), nn.Dropout(dropout),

        )


    def forward(self, cls_tok, patch_tok):


        h = self.norm1(cls_tok)

        h, _ = self.self_a(h, h, h, need_weights=False)

        cls_tok = cls_tok + h


        h = self.norm2(cls_tok)

        h, _ = self.cross(h, patch_tok, patch_tok, need_weights=False)

        cls_tok = cls_tok + h


        cls_tok = cls_tok + self.mlp(self.norm3(cls_tok))

        return cls_tok


class MaskTransformerDecoder(nn.Module):

    def __init__(self, num_classes, dim, num_heads, depth, mlp_ratio=4.0, dropout=0.0):

        super().__init__()

        self.cls_tokens = nn.Parameter(torch.randn(1, num_classes, dim) * 0.02)

        self.layers = nn.ModuleList(

            [MaskDecoderLayer(dim, num_heads, mlp_ratio, dropout) for _ in range(depth)]

        )

        self.norm = nn.LayerNorm(dim)


    def forward(self, patch_tok, h, w):


        B = patch_tok.shape[0]

        cls = self.cls_tokens.expand(B, -1, -1)

        for layer in self.layers:

            cls = layer(cls, patch_tok)

        cls = self.norm(cls)


        patch_tok = self.norm(patch_tok) if False else patch_tok

        masks = cls @ patch_tok.transpose(1, 2)

        return masks.reshape(B, cls.shape[1], h, w)


class Segmenter(nn.Module):

    def __init__(

        self,

        in_channels:  int   = 1,

        out_channels: int   = 1,

        patch_size:   int   = 16,

        embed_dim:    int   = 256,

        encoder_depth: int  = 12,

        decoder_depth: int  = 2,

        num_heads:    int   = 8,

        mlp_ratio:    float = 4.0,

        dropout:      float = 0.0,

        decoder:      str   = "mask",

    ):

        super().__init__()

        self.patch_size   = patch_size

        self.embed_dim    = embed_dim

        self.decoder_type = decoder

        self.out_channels = out_channels


        self.patch_embed = PatchEmbed(in_channels, embed_dim, patch_size)

        self.encoder = nn.ModuleList(

            [TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout) for _ in range(encoder_depth)]

        )

        self.enc_norm = nn.LayerNorm(embed_dim)


        if decoder == "mask":

            self.mask_decoder = MaskTransformerDecoder(

                num_classes=out_channels,

                dim=embed_dim,

                num_heads=num_heads,

                depth=decoder_depth,

                mlp_ratio=mlp_ratio,

                dropout=dropout,

            )

        else:

            self.linear_head = nn.Conv2d(embed_dim, out_channels, 1)


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

        for blk in self.encoder:

            tokens = blk(tokens)

        tokens = self.enc_norm(tokens)


        if self.decoder_type == "mask":

            logits = self.mask_decoder(tokens, h, w)

        else:

            feat   = tokens.transpose(1, 2).reshape(B, self.embed_dim, h, w)

            logits = self.linear_head(feat)


        return F.interpolate(logits, size=(H, W), mode="bilinear", align_corners=False)

