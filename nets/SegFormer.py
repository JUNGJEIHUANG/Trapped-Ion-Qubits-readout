


import torch

import torch.nn as nn

import torch.nn.functional as F


class EfficientAttn(nn.Module):

    def __init__(self, dim, num_heads, sr_ratio=1, dropout=0.0):

        super().__init__()

        assert dim % num_heads == 0

        self.num_heads = num_heads

        self.head_dim  = dim // num_heads

        self.scale     = self.head_dim ** -0.5


        self.q    = nn.Linear(dim, dim)

        self.kv   = nn.Linear(dim, dim * 2)

        self.proj = nn.Linear(dim, dim)

        self.drop = nn.Dropout(dropout)


        self.sr_ratio = sr_ratio

        if sr_ratio > 1:

            self.sr   = nn.Conv2d(dim, dim, sr_ratio, stride=sr_ratio)

            self.norm = nn.LayerNorm(dim)


    def forward(self, x, H, W):

        B, N, C = x.shape

        Nh, Dh = self.num_heads, self.head_dim


        q = self.q(x).reshape(B, N, Nh, Dh).transpose(1, 2)


        if self.sr_ratio > 1:

            x_ = x.transpose(1, 2).reshape(B, C, H, W)

            x_ = self.sr(x_).reshape(B, C, -1).transpose(1, 2)

            x_ = self.norm(x_)

        else:

            x_ = x


        kv = self.kv(x_).reshape(B, -1, 2, Nh, Dh).permute(2, 0, 3, 1, 4)

        k, v = kv[0], kv[1]


        attn = (q @ k.transpose(-2, -1)) * self.scale

        attn = self.drop(attn.softmax(dim=-1))


        out = (attn @ v).transpose(1, 2).reshape(B, N, C)

        return self.proj(out)


class MixFFN(nn.Module):

    def __init__(self, dim, expand=4, dropout=0.0):

        super().__init__()

        hid = int(dim * expand)

        self.fc1   = nn.Linear(dim, hid)

        self.dw    = nn.Conv2d(hid, hid, 3, padding=1, groups=hid)

        self.act   = nn.GELU()

        self.fc2   = nn.Linear(hid, dim)

        self.drop  = nn.Dropout(dropout)


    def forward(self, x, H, W):

        B, N, C = x.shape

        x = self.fc1(x)

        x = x.transpose(1, 2).reshape(B, -1, H, W)

        x = self.dw(x).reshape(B, -1, N).transpose(1, 2)

        x = self.drop(self.act(x))

        return self.drop(self.fc2(x))


class MixBlock(nn.Module):

    def __init__(self, dim, num_heads, sr_ratio=1, mlp_expand=4, dropout=0.0):

        super().__init__()

        self.norm1 = nn.LayerNorm(dim)

        self.attn  = EfficientAttn(dim, num_heads, sr_ratio, dropout)

        self.norm2 = nn.LayerNorm(dim)

        self.ffn   = MixFFN(dim, mlp_expand, dropout)


    def forward(self, x, H, W):

        x = x + self.attn(self.norm1(x), H, W)

        x = x + self.ffn(self.norm2(x), H, W)

        return x


class OverlapPatchEmbed(nn.Module):

    def __init__(self, in_channels, embed_dim, kernel_size, stride):

        super().__init__()

        pad = kernel_size // 2

        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size, stride=stride, padding=pad)

        self.norm = nn.LayerNorm(embed_dim)


    def forward(self, x):

        x = self.proj(x)

        B, D, H, W = x.shape

        x = x.flatten(2).transpose(1, 2)

        return self.norm(x), H, W


class MiTStage(nn.Module):

    def __init__(self, in_ch, embed_dim, num_heads, depth, sr_ratio,

                 kernel_size, stride, mlp_expand=4, dropout=0.0):

        super().__init__()

        self.patch_embed = OverlapPatchEmbed(in_ch, embed_dim, kernel_size, stride)

        self.blocks = nn.ModuleList(

            [MixBlock(embed_dim, num_heads, sr_ratio, mlp_expand, dropout) for _ in range(depth)]

        )

        self.norm = nn.LayerNorm(embed_dim)


    def forward(self, x):

        tokens, H, W = self.patch_embed(x)

        for blk in self.blocks:

            tokens = blk(tokens, H, W)

        tokens = self.norm(tokens)

        B, _, C = tokens.shape

        feat = tokens.transpose(1, 2).reshape(B, C, H, W)

        return feat


class MLPHead(nn.Module):

    def __init__(self, in_dim, decoder_dim):

        super().__init__()

        self.proj = nn.Linear(in_dim, decoder_dim)


    def forward(self, x):

        B, C, H, W = x.shape

        x = x.flatten(2).transpose(1, 2)

        return self.proj(x).transpose(1, 2).reshape(B, -1, H, W)


class SegFormer(nn.Module):


    def __init__(

        self,

        in_channels:  int   = 1,

        out_channels: int   = 1,

        embed_dims:   tuple = (32, 64, 128, 256),

        num_heads:    tuple = (1, 2, 4, 8),

        depths:       tuple = (2, 2, 2, 2),

        sr_ratios:    tuple = (8, 4, 2, 1),

        mlp_expand:   int   = 4,

        decoder_dim:  int   = 256,

        dropout:      float = 0.0,

    ):

        super().__init__()

        dims = embed_dims


        self.stage1 = MiTStage(in_channels, dims[0], num_heads[0], depths[0], sr_ratios[0],

                                kernel_size=7, stride=4, mlp_expand=mlp_expand, dropout=dropout)

        self.stage2 = MiTStage(dims[0], dims[1], num_heads[1], depths[1], sr_ratios[1],

                                kernel_size=3, stride=2, mlp_expand=mlp_expand, dropout=dropout)

        self.stage3 = MiTStage(dims[1], dims[2], num_heads[2], depths[2], sr_ratios[2],

                                kernel_size=3, stride=2, mlp_expand=mlp_expand, dropout=dropout)

        self.stage4 = MiTStage(dims[2], dims[3], num_heads[3], depths[3], sr_ratios[3],

                                kernel_size=3, stride=2, mlp_expand=mlp_expand, dropout=dropout)


        self.head1 = MLPHead(dims[0], decoder_dim)

        self.head2 = MLPHead(dims[1], decoder_dim)

        self.head3 = MLPHead(dims[2], decoder_dim)

        self.head4 = MLPHead(dims[3], decoder_dim)


        self.fuse = nn.Sequential(

            nn.Conv2d(decoder_dim * 4, decoder_dim, 1, bias=False),

            nn.BatchNorm2d(decoder_dim),

            nn.ReLU(inplace=True),

        )

        self.out_conv = nn.Conv2d(decoder_dim, out_channels, 1)


    def forward(self, x):

        B, C, H, W = x.shape


        f1 = self.stage1(x)

        f2 = self.stage2(f1)

        f3 = self.stage3(f2)

        f4 = self.stage4(f3)


        target = f1.shape[2:]


        p1 = self.head1(f1)

        p2 = F.interpolate(self.head2(f2), size=target, mode="bilinear", align_corners=False)

        p3 = F.interpolate(self.head3(f3), size=target, mode="bilinear", align_corners=False)

        p4 = F.interpolate(self.head4(f4), size=target, mode="bilinear", align_corners=False)


        fused = self.fuse(torch.cat([p1, p2, p3, p4], dim=1))

        out   = self.out_conv(fused)

        return F.interpolate(out, size=(H, W), mode="bilinear", align_corners=False)

