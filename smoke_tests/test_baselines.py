"""Shape/gradient-flow smoke test for the App. F coordinate-aware baselines

(nets/DETRQuery.py, nets/SetTransformer.py), including a check that their

DW-UNet backbone submodule sits at the same state-dict key prefix as

DWNetV2_unet's, which train_main.py::load_pretrained_weights relies on.

"""

import _pathsetup  # noqa: F401

import torch

from nets.DETRQuery import DETRStyleQuery

from nets.SetTransformer import SetTransformerReadout

from nets.DWNetV2_unet import DWNetV2_unet


def _run_forward_backward(model, image, coords):

    model.train()

    out = model(image, site_coords=coords)

    assert out["bright_logit"].shape == coords.shape[:2]

    assert out["pred_coords"].shape == coords.shape

    assert out["mask_logits"].shape[0] == coords.shape[0]

    loss = out["bright_logit"].sum() + out["mask_logits"].sum()

    loss.backward()

    return out


def test_detr_style_query():

    torch.manual_seed(0)

    B, H, W, K = 2, 88, 456, 20

    image = torch.randn(B, 1, H, W)

    coords = torch.rand(B, K, 2)

    coords[..., 0] *= W - 1

    coords[..., 1] *= H - 1


    model = DETRStyleQuery(num_ions=300)

    _run_forward_backward(model, image, coords)


def test_set_transformer():

    torch.manual_seed(0)

    B, H, W, K = 2, 88, 456, 20

    image = torch.randn(B, 1, H, W)

    coords = torch.rand(B, K, 2)

    coords[..., 0] *= W - 1

    coords[..., 1] *= H - 1


    model = SetTransformerReadout(num_ions=300)

    _run_forward_backward(model, image, coords)


def test_backbone_key_prefix_matches_dwnetv2_unet():

    detr_keys = {k for k in DETRStyleQuery().state_dict() if k.startswith("backbone.")}

    set_keys = {k for k in SetTransformerReadout().state_dict() if k.startswith("backbone.")}

    ref_keys = {k for k in DWNetV2_unet(pre_trained=None).state_dict() if k.startswith("backbone.")}

    assert detr_keys == ref_keys

    assert set_keys == ref_keys

    assert len(ref_keys) > 0


if __name__ == "__main__":

    test_detr_style_query()

    test_set_transformer()

    test_backbone_key_prefix_matches_dwnetv2_unet()

    print("PASS: test_baselines")

