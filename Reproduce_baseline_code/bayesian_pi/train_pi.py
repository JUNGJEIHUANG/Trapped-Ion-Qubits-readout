"""Train the PI-Network to match the exact Bayesian posterior.

Zhou et al.: "The network is trained by minimizing the Kullback-Leibler (KL)
divergence loss over a sufficiently broad sampling of parameters
(theta_f, theta_g, l, N) covering the experimental range."

Training data is generated on the fly, so no dataset is needed: the target is
the analytic posterior of Eq. (2), which we can evaluate exactly. The network
is therefore distilling a known function, and the honest success criterion is
agreement with that function -- reported here as mean KL and as the mean
absolute error in the posterior mean ``l-bar``.

Usage
-----
    python -m Reproduce_baseline_code.bayesian_pi.train_pi --steps 4000 --out pi_net.pt
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import torch

from .pi_network import PINetwork, PINetworkConfig, make_training_batch


def train(
    steps: int = 4000,
    batch_size: int = 64,
    n_shots_choices=(25, 50, 100, 200, 400),
    lr: float = 3e-4,
    n_grid: int = 101,
    seed: int = 0,
    device: str = "cpu",
    log_every: int = 500,
):
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    net = PINetwork(PINetworkConfig(n_grid=n_grid)).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)

    net.train()
    t0 = time.time()
    for step in range(1, steps + 1):
        n_shots = int(rng.choice(n_shots_choices))
        S, AUX, TGT = make_training_batch(batch_size, n_shots, rng, n_grid=n_grid)

        s = torch.as_tensor(S, device=device)
        aux = torch.as_tensor(AUX, device=device)
        tgt = torch.as_tensor(TGT, device=device)

        log_p = net(s, aux)
        # KL(target || net); the target-entropy term is constant in the
        # parameters but is kept so the printed loss is a true divergence and
        # can be read as "how far from exact", not just a relative score.
        loss = torch.sum(tgt * (torch.log(tgt.clamp_min(1e-12)) - log_p), dim=-1).mean()

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
        opt.step()
        sched.step()

        if step % log_every == 0 or step == 1:
            with torch.no_grad():
                grid = net.l_grid
                l_net = (log_p.exp() * grid).sum(-1)
                l_tgt = (tgt * grid).sum(-1)
                mae = (l_net - l_tgt).abs().mean().item()
            print(
                f"step {step:5d}/{steps}  KL={loss.item():.5f}  "
                f"MAE(l-bar)={mae:.5f}  ({time.time() - t0:.1f}s)"
            )
    return net


def evaluate(net, n_batches: int = 20, batch_size: int = 64, n_shots: int = 200, seed: int = 123, device="cpu"):
    """Agreement with the exact posterior, on parameters never seen in training."""
    rng = np.random.default_rng(seed)
    net.eval()
    kls, maes = [], []
    with torch.no_grad():
        for _ in range(n_batches):
            S, AUX, TGT = make_training_batch(batch_size, n_shots, rng, n_grid=net.config.n_grid)
            s = torch.as_tensor(S, device=device)
            aux = torch.as_tensor(AUX, device=device)
            tgt = torch.as_tensor(TGT, device=device)
            log_p = net(s, aux)
            kls.append(torch.sum(tgt * (torch.log(tgt.clamp_min(1e-12)) - log_p), dim=-1).mean().item())
            grid = net.l_grid
            maes.append(((log_p.exp() * grid).sum(-1) - (tgt * grid).sum(-1)).abs().mean().item())
    return {"kl": float(np.mean(kls)), "mae_l": float(np.mean(maes))}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--n-grid", type=int, default=101)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default="pi_net.pt")
    args = ap.parse_args()

    net = train(
        steps=args.steps,
        batch_size=args.batch_size,
        lr=args.lr,
        n_grid=args.n_grid,
        seed=args.seed,
        device=args.device,
    )
    print("held-out agreement with exact posterior:", evaluate(net, device=args.device))
    torch.save({"state_dict": net.state_dict(), "config": net.config.__dict__}, args.out)
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
