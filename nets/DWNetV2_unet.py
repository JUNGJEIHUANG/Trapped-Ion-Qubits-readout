import logging

import math

import sys


import torch

import torch.nn as nn

from torch.nn.functional import interpolate

from torch.nn import functional as F

from nets.DWNetV2 import DWNetV2, InvertedResidual

from nets.site_dia import SiteDIAHead


class DWNetV2_unet(nn.Module):

    def __init__(

        self,

        pre_trained,

        mode="train",

        enable_site_dia=False,

        num_ions=300,

        dia_token_dim=256,

        dia_num_heads=8,

        dia_num_ion_attn_layers=1,

        dia_tok_radius=6,

        dia_tok_num_select=16,

        dia_tok_offset_max_px=3.0,

        dia_tau_start=0.8,

        dia_tau_end=0.1,

        dia_max_coord_offset=3.0,

        dia_use_global_cross_attn=False,

        dia_disable_offset=False,

        dia_disable_psf_mask=False,

        dia_disable_coord_head=False,

    ):

        super(DWNetV2_unet, self).__init__()


        self.mode = mode

        self.enable_site_dia = enable_site_dia

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


        if self.enable_site_dia:


            self.site_dia = SiteDIAHead(

                num_ions=num_ions,

                in_channels=17,

                token_dim=dia_token_dim,

                num_heads=dia_num_heads,

                num_ion_attn_layers=dia_num_ion_attn_layers,

                tok_radius=dia_tok_radius,

                tok_num_select=dia_tok_num_select,

                tok_offset_max_px=dia_tok_offset_max_px,

                tau_start=dia_tau_start,

                tau_end=dia_tau_end,

                max_coord_offset=dia_max_coord_offset,

                use_global_cross_attn=dia_use_global_cross_attn,

                disable_offset=dia_disable_offset,

                disable_psf_mask=dia_disable_psf_mask,

                disable_coord_head=dia_disable_coord_head,

            )


        self._init_weights()


        if pre_trained is not None:


            full_state_dict = torch.load(pre_trained, map_location="cuda")


            backbone_state_dict = {}

            for key, value in full_state_dict.items():

                if key.startswith("backbone."):

                    new_key = key[len("backbone."):]

                    backbone_state_dict[new_key] = value


            self.backbone.load_state_dict(backbone_state_dict)


        else:

            pass


    def forward(self, x, site_coords=None, return_dict=False):

        raw_input = x

        for n in range(0, 2):

            x = self.backbone.features[n](x)

        x1 = x

        logging.debug((x1.shape, "x1"))


        for n in range(2, 4):

            x = self.backbone.features[n](x)

        x2 = x

        logging.debug((x2.shape, "x2"))


        for n in range(4, 7):

            x = self.backbone.features[n](x)

        x3 = x

        logging.debug((x3.shape, "x3"))


        for n in range(7, 14):

            x = self.backbone.features[n](x)

        x4 = x

        logging.debug((x4.shape, "x4"))


        for n in range(14, 19):

            x = self.backbone.features[n](x)

        x5 = x

        logging.debug((x5.shape, "x5"))


        decoder_feature0 = self.dconv1(x)

        decoder_feature0 = F.interpolate(

            decoder_feature0, size=x4.shape[2:], mode="bilinear", align_corners=True

        )

        up1 = torch.cat([x4, decoder_feature0], dim=1)

        up1 = self.invres1(up1)

        logging.debug((up1.shape, "up1"))


        decoder_feature1 = self.dconv2(up1)

        decoder_feature1 = F.interpolate(

            decoder_feature1, size=x3.shape[2:], mode="bilinear", align_corners=True

        )

        up2 = torch.cat([x3, decoder_feature1], dim=1)

        up2 = self.invres2(up2)

        logging.debug((up2.shape, "up2"))


        decoder_feature2 = self.dconv3(up2)

        decoder_feature2 = F.interpolate(

            decoder_feature2, size=x2.shape[2:], mode="bilinear", align_corners=True

        )

        up3 = torch.cat([x2, decoder_feature2], dim=1)

        up3 = self.invres3(up3)

        logging.debug((up3.shape, "up3"))


        decoder_feature3 = self.dconv4(up3)

        decoder_feature3 = F.interpolate(

            decoder_feature3, size=x1.shape[2:], mode="bilinear", align_corners=True

        )

        up4 = torch.cat([x1, decoder_feature3], dim=1)

        up4 = self.invres4(up4)

        logging.debug((up4.shape, "up4"))


        x = self.conv_last(up4)

        logging.debug((x.shape, "conv_last"))


        mask_logits = self.conv_score(x)

        logging.debug((mask_logits.shape, "conv_score"))


        if self.enable_site_dia or return_dict:

            out = {"mask_logits": mask_logits}

            if self.enable_site_dia:

                if site_coords is None:

                    raise ValueError("site_coords must be provided when enable_site_dia=True")

                out.update(self.site_dia(raw_input, up4, site_coords))

            return out


        return mask_logits


    def set_temperature_progress(self, progress):

        if self.enable_site_dia:

            self.site_dia.set_temperature_progress(progress)


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


if __name__ == "__main__":


    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

    net = DWNetV2_unet(pre_trained=None)

    net(torch.randn(1, 1, 88, 456))

