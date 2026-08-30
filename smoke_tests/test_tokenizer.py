"""Shape/gradient-flow smoke test for nets/site_tokenizer.py.

Not a correctness or accuracy test -- uses random tensors and only checks that

the forward pass produces the expected shapes and that gradients reach every

learnable component, including through the Gumbel/straight-through Top-16 gate.

"""

import _pathsetup  # noqa: F401

import torch

from nets.site_tokenizer import AdaptiveTop16Tokenizer, GlobalCrossAttnTokenizer


def test_adaptive_top16_tokenizer():

    torch.manual_seed(0)

    B, C, H, W, K = 2, 16, 88, 456, 10


    feat = torch.randn(B, C, H, W, requires_grad=True)

    coords = torch.rand(B, K, 2)

    coords[..., 0] *= W - 1

    coords[..., 1] *= H - 1


    tok = AdaptiveTop16Tokenizer(in_channels=C)

    tok.train()

    out, aux = tok(feat, coords)

    assert out.shape == (B, K, 16 * C)

    assert aux["tok_offset"].shape == (B, K, 2)


    out.sum().backward()

    assert tok.g_sel[0].weight.grad is not None and tok.g_sel[0].weight.grad.abs().sum() > 0

    assert tok.g_delta[-1].weight.grad is not None and tok.g_delta[-1].weight.grad.abs().sum() > 0

    assert tok.psf_mask.grad is not None and tok.psf_mask.grad.abs().sum() > 0

    assert feat.grad is not None


    tok.eval()

    out_eval, _ = tok(feat, coords)

    assert out_eval.shape == (B, K, 16 * C)


    tok_ablated = AdaptiveTop16Tokenizer(in_channels=C, disable_offset=True, disable_psf_mask=True)

    out_ablated, aux_ablated = tok_ablated(feat, coords)

    assert torch.all(aux_ablated["tok_offset"] == 0)

    assert out_ablated.shape == (B, K, 16 * C)


def test_global_cross_attn_tokenizer():

    torch.manual_seed(0)

    B, C, H, W, K = 2, 16, 88, 456, 10

    feat = torch.randn(B, C, H, W, requires_grad=True)

    coords = torch.rand(B, K, 2)

    coords[..., 0] *= W - 1

    coords[..., 1] *= H - 1


    tok = GlobalCrossAttnTokenizer(in_channels=C, out_dim=256, num_heads=8)

    out, _ = tok(feat, coords)

    assert out.shape == (B, K, 256)

    out.sum().backward()

    assert feat.grad is not None


if __name__ == "__main__":

    test_adaptive_top16_tokenizer()

    test_global_cross_attn_tokenizer()

    print("PASS: test_tokenizer")

