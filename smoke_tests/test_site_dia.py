"""Shape/gradient-flow smoke test for the full Site-AIT head (nets/site_dia.py)

wired through nets/DWNetV2_unet.py, across every Table VI architectural

ablation flag combination.

"""

import _pathsetup  # noqa: F401

import torch

from nets.DWNetV2_unet import DWNetV2_unet


CONFIGS = {

    "full": {},

    "global_cross_attn": {"dia_use_global_cross_attn": True},

    "no_self_attn": {"dia_num_ion_attn_layers": 0},

    "no_tok_offsets": {"dia_disable_offset": True},

    "no_psf_mask": {"dia_disable_psf_mask": True},

    "no_coord_constraint": {"dia_disable_coord_head": True},

}


def test_all_architectural_variants():

    torch.manual_seed(0)

    B, H, W, K = 2, 88, 456, 20


    image = torch.randn(B, 1, H, W)

    coords = torch.rand(B, K, 2)

    coords[..., 0] *= W - 1

    coords[..., 1] *= H - 1


    for name, kwargs in CONFIGS.items():

        model = DWNetV2_unet(pre_trained=None, enable_site_dia=True, num_ions=300, **kwargs)

        model.train()

        out = model(image, site_coords=coords)


        assert out["bright_logit"].shape == (B, K)

        assert out["pred_coords"].shape == (B, K, 2)

        assert out["exist_logit"].shape == (B, K)

        assert out["mask_logits"].shape == (B, 1, H, W)


        loss = out["bright_logit"].sum() + out["pred_coords"].sum() + out["exist_logit"].sum()

        loss.backward()


        model.set_temperature_progress(0.5)  # must not raise for any variant


def test_no_coord_constraint_freezes_coords():

    torch.manual_seed(0)

    B, H, W, K = 1, 88, 456, 5

    image = torch.randn(B, 1, H, W)

    coords = torch.rand(B, K, 2)

    coords[..., 0] *= W - 1

    coords[..., 1] *= H - 1


    model = DWNetV2_unet(pre_trained=None, enable_site_dia=True, num_ions=300, dia_disable_coord_head=True)

    model.eval()

    out = model(image, site_coords=coords)

    assert torch.allclose(out["pred_coords"], coords)


if __name__ == "__main__":

    test_all_architectural_variants()

    test_no_coord_constraint_freezes_coords()

    print("PASS: test_site_dia")

