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


class StandardUNet(nn.Module):


    def __init__(self, in_channels=1, out_channels=1, base_channels=32):

        super().__init__()

        c = base_channels

        self.enc1 = DoubleConv(in_channels, c)

        self.enc2 = DoubleConv(c, c * 2)

        self.enc3 = DoubleConv(c * 2, c * 4)

        self.enc4 = DoubleConv(c * 4, c * 8)

        self.pool = nn.MaxPool2d(2)

        self.bottleneck = DoubleConv(c * 8, c * 16)


        self.up4 = nn.ConvTranspose2d(c * 16, c * 8, 2, stride=2)

        self.dec4 = DoubleConv(c * 16, c * 8)

        self.up3 = nn.ConvTranspose2d(c * 8, c * 4, 2, stride=2)

        self.dec3 = DoubleConv(c * 8, c * 4)

        self.up2 = nn.ConvTranspose2d(c * 4, c * 2, 2, stride=2)

        self.dec2 = DoubleConv(c * 4, c * 2)

        self.up1 = nn.ConvTranspose2d(c * 2, c, 2, stride=2)

        self.dec1 = DoubleConv(c * 2, c)

        self.out = nn.Conv2d(c, out_channels, 1)


    @staticmethod

    def _align(x, ref):

        if x.shape[2:] != ref.shape[2:]:

            x = F.interpolate(

                x, size=ref.shape[2:], mode="bilinear", align_corners=False

            )

        return x


    def forward(self, x):

        e1 = self.enc1(x)

        e2 = self.enc2(self.pool(e1))

        e3 = self.enc3(self.pool(e2))

        e4 = self.enc4(self.pool(e3))

        b = self.bottleneck(self.pool(e4))


        d4 = self._align(self.up4(b), e4)

        d4 = self.dec4(torch.cat([e4, d4], dim=1))

        d3 = self._align(self.up3(d4), e3)

        d3 = self.dec3(torch.cat([e3, d3], dim=1))

        d2 = self._align(self.up2(d3), e2)

        d2 = self.dec2(torch.cat([e2, d2], dim=1))

        d1 = self._align(self.up1(d2), e1)

        d1 = self.dec1(torch.cat([e1, d1], dim=1))

        return self.out(d1)

