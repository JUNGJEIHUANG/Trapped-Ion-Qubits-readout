import math

import torch

import torch.nn as nn

from torch.nn import functional as F

from nets.DWNetV2 import DWNetV2, InvertedResidual


class DWUNetBackboneDecoder(nn.Module):

    """Shared depthwise-separable U-Net encoder/decoder.

    Site-AIT and its two coordinate-aware baselines (DETR-style Query, Set

    Transformer) all use "the same DW-UNet backbone" per the paper's App. F

    fairness protocol. Subclassing this base (rather than composing an inner

    instance) keeps every submodule at the top level of the owning model's

    state dict, so pretrained-checkpoint loading via

    train_main.py::load_pretrained_weights (which matches "backbone."-prefixed

    keys) works identically for DWNetV2_unet, DETRStyleQuery, and

    SetTransformerReadout.

    """


    def __init__(self):

        super().__init__()

        self.backbone = DWNetV2()


        self.dconv1 = nn.ConvTranspose2d(1280, 96, 4, padding=1, stride=2)

        self.invres1 = InvertedResidual(192, 96, 1, 6)


        self.dconv2 = nn.ConvTranspose2d(96, 32, 4, padding=1, stride=2)

        self.invres2 = InvertedResidual(64, 32, 1, 6)


        self.dconv3 = nn.ConvTranspose2d(32, 24, 4, padding=1, stride=2)

        self.invres3 = InvertedResidual(48, 24, 1, 6)


        self.dconv4 = nn.ConvTranspose2d(24, 16, 4, padding=1, stride=2)

        self.invres4 = InvertedResidual(32, 16, 1, 6)


        self.conv_last = nn.Conv2d(16, 3, 1)

        self.conv_score = nn.Conv2d(3, 1, 1)


    def extract_features(self, x):

        """Returns (raw_input, up4, mask_logits): up4 is the 16-channel decoder

        feature field (C_F=16, Sec. IV.B of the paper); mask_logits is the

        diagnostic dense head, not part of the site-token readout path."""

        raw_input = x

        for n in range(0, 2):

            x = self.backbone.features[n](x)

        x1 = x

        for n in range(2, 4):

            x = self.backbone.features[n](x)

        x2 = x

        for n in range(4, 7):

            x = self.backbone.features[n](x)

        x3 = x

        for n in range(7, 14):

            x = self.backbone.features[n](x)

        x4 = x

        for n in range(14, 19):

            x = self.backbone.features[n](x)

        x5 = x


        decoder_feature0 = self.dconv1(x5)

        decoder_feature0 = F.interpolate(

            decoder_feature0, size=x4.shape[2:], mode="bilinear", align_corners=True

        )

        up1 = torch.cat([x4, decoder_feature0], dim=1)

        up1 = self.invres1(up1)


        decoder_feature1 = self.dconv2(up1)

        decoder_feature1 = F.interpolate(

            decoder_feature1, size=x3.shape[2:], mode="bilinear", align_corners=True

        )

        up2 = torch.cat([x3, decoder_feature1], dim=1)

        up2 = self.invres2(up2)


        decoder_feature2 = self.dconv3(up2)

        decoder_feature2 = F.interpolate(

            decoder_feature2, size=x2.shape[2:], mode="bilinear", align_corners=True

        )

        up3 = torch.cat([x2, decoder_feature2], dim=1)

        up3 = self.invres3(up3)


        decoder_feature3 = self.dconv4(up3)

        decoder_feature3 = F.interpolate(

            decoder_feature3, size=x1.shape[2:], mode="bilinear", align_corners=True

        )

        up4 = torch.cat([x1, decoder_feature3], dim=1)

        up4 = self.invres4(up4)


        feat = self.conv_last(up4)

        mask_logits = self.conv_score(feat)


        return raw_input, up4, mask_logits


    def _init_weights(self):

        for m in self.modules():

            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):

                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels

                m.weight.data.normal_(0, math.sqrt(2.0 / n))

                if m.bias is not None:

                    m.bias.data.zero_()

            elif isinstance(m, nn.BatchNorm2d):

                m.weight.data.fill_(1)

                m.bias.data.zero_()

            elif isinstance(m, nn.Linear):

                m.weight.data.normal_(0, 0.01)

                m.bias.data.zero_()

