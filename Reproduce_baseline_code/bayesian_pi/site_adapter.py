"""Site-AIT adaptation of Zhou et al. (PRL 137, 013601, 2026).

Why an adapter is needed
------------------------
Zhou et al. estimate ``l``, the bright-state *occupation probability* of a
qubit, from ``N`` repeated shots of ROI-integrated photon counts. The output is
a continuous posterior with a calibrated uncertainty. Site-AIT classifies each
of 300 sites as bright or dark from a *single* frame.

These are different estimands. Feeding the paper's estimator one shot and
thresholding its posterior mean would guarantee a poor number and would tell a
reader nothing -- it would be a strawman, not a baseline. Two honest modes are
therefore provided, and the protocol document specifies that both be reported.

Mode A -- ensemble (``EnsembleReadout``)
    The paper's own task, unchanged. Used to validate the reproduction against
    the published numbers, and to compare on the paper's own ground: given a
    block of frames prepared in the same state, how well is ``l`` recovered?

Mode B -- single-frame, per-site (``SiteBayesianReadout``)
    The strongest faithful reading of the method under Site-AIT's protocol.
    Two things carry over exactly:

      * the *weakly anchored* calibration -- only ``g`` is anchored, from
        all-dark frames; ``f`` is inferred self-consistently by EM on unlabelled
        mixed-state frames, never from the labels;
      * the log-likelihood-ratio decision statistic ``s_k = log[f(n_k)/g(n_k)]``,
        which is what the paper's posterior reduces to for a single observation.

    The permutation-invariant network is redirected from "invariant over shots"
    to "invariant over sites within one frame". This is the natural transfer:
    in both cases the network consumes an unordered collection of LLRs and the
    exact posterior depends on that collection only through pooled statistics.
    It also gives the method a genuine register-level channel, which is the
    fair way to test it against Site-AIT's cross-site attention.

Mode B is the row that belongs in Table 1. Mode A belongs in the supplementary.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from ..common.metrics import classification_fidelity
from .bayes_em import posterior_over_l, weakly_anchored_em
from .distributions import SuperPoissonian, Theta, histogram_overlap

__all__ = ["ROIExtractor", "SiteBayesianReadout", "EnsembleReadout"]


class ROIExtractor:
    """Integrates photon counts in a circular ROI around each calibrated site.

    Zhou et al. sum "over all pixels in the ROI" per shot. The ROI radius is an
    experiment-specific choice; here it is selected on the validation split,
    exactly as Site-AIT selects its own evaluation-disk radius
    (``app:mask_to_site_protocol``), so neither method gets a hand-tuned
    advantage.
    """

    def __init__(self, centers: np.ndarray, shape: tuple, radius: float):
        self.centers = np.asarray(centers, dtype=float)
        self.shape = shape
        self.radius = float(radius)
        self._masks = self._build()

    def _build(self):
        H, W = self.shape
        r = int(np.ceil(self.radius))
        masks = []
        for r0, c0 in self.centers:
            r_lo, r_hi = int(max(0, round(r0) - r)), int(min(H, round(r0) + r + 1))
            c_lo, c_hi = int(max(0, round(c0) - r)), int(min(W, round(c0) + r + 1))
            rr, cc = np.mgrid[r_lo:r_hi, c_lo:c_hi]
            inside = ((rr - r0) ** 2 + (cc - c0) ** 2) <= self.radius**2
            masks.append((slice(r_lo, r_hi), slice(c_lo, c_hi), inside))
        return masks

    def counts(self, frames: np.ndarray) -> np.ndarray:
        """(n_frames, n_sites) ROI-integrated counts.

        Counts are clipped at zero: the frames are background-subtracted, so a
        dark ROI can integrate to a small negative value, which no photon-count
        distribution can represent.
        """
        out = np.empty((frames.shape[0], len(self._masks)), dtype=float)
        for k, (rs, cs, inside) in enumerate(self._masks):
            out[:, k] = frames[:, rs, cs][:, inside].sum(axis=1)
        return np.clip(out, 0.0, None)


@dataclass
class _SiteModel:
    theta_g: Theta
    theta_f: Theta
    threshold: float
    overlap: float


class SiteBayesianReadout:
    """Mode B: weakly anchored Bayesian per-site readout from a single frame.

    Calibration protocol
    --------------------
    1. ``g`` is anchored on all-dark training frames -- the direct analogue of
       the paper's "background exposures of empty tweezers". No labels beyond
       the frame-level all-dark flag are used.
    2. ``f`` is inferred by the weakly anchored EM iteration on *unlabelled*
       mixed-state training frames, per site. Site labels are never consumed.
    3. The decision threshold on the LLR is selected on the validation split.
       This is the only step that uses site labels, and it mirrors the
       threshold selection every other baseline in Table 1 receives.

    Optionally, a PI-Network trained over sites can replace the independent
    per-site decision with a register-coupled one (``attach_pi_network``).
    """

    def __init__(self, roi: ROIExtractor, n_grid: int = 501, max_iter: int = 10):
        self.roi = roi
        self.n_grid = n_grid
        self.max_iter = max_iter
        self.n_sites = roi.centers.shape[0]
        self.models_: list[_SiteModel] = []
        self.pi_network_ = None

    def calibrate(
        self,
        dark_frames: np.ndarray,
        mixed_frames: np.ndarray,
        val_frames: np.ndarray,
        val_labels: np.ndarray,
        thresholds: Optional[Sequence[float]] = None,
        verbose: bool = False,
    ) -> "SiteBayesianReadout":
        counts_dark = self.roi.counts(dark_frames)
        counts_mixed = self.roi.counts(mixed_frames)
        counts_val = self.roi.counts(val_frames)

        self.models_ = []
        for k in range(self.n_sites):
            theta_g = SuperPoissonian.fit_moments(counts_dark[:, k])

            # Weakly anchored EM: g is fixed, f is inferred self-consistently.
            res = weakly_anchored_em(
                counts_mixed[:, k], theta_g, n_grid=self.n_grid, max_iter=self.max_iter
            )
            theta_f = res.theta_f

            # EM can land on f ~= g when the site is nearly always dark; in that
            # degenerate case the LLR carries no signal. Fall back to a
            # moment-matched split of the upper tail so the site still yields a
            # usable statistic instead of a constant.
            mu_f, _ = SuperPoissonian.moments(theta_f)
            mu_g, _ = SuperPoissonian.moments(theta_g)
            if mu_f <= mu_g * 1.05:
                hi = counts_mixed[:, k] >= np.quantile(counts_mixed[:, k], 0.5)
                theta_f = SuperPoissonian.fit_moments(counts_mixed[hi, k])

            s_val = self._llr(counts_val[:, k], theta_f, theta_g)
            grid = thresholds if thresholds is not None else np.quantile(
                s_val, np.linspace(0.01, 0.99, 99)
            )
            y_val = val_labels[:, k].astype(bool)
            fids = [classification_fidelity(y_val, s_val >= t) for t in grid]
            thr = float(grid[int(np.argmax(fids))])

            self.models_.append(
                _SiteModel(theta_g, theta_f, thr, histogram_overlap(theta_f, theta_g))
            )
            if verbose and (k % 50 == 0 or k == self.n_sites - 1):
                print(
                    f"  site {k:4d}/{self.n_sites}: overlap={self.models_[-1].overlap:.3f} "
                    f"thr={thr:+.3f} val_fidelity={max(fids):.5f}"
                )
        return self

    @staticmethod
    def _llr(counts: np.ndarray, theta_f: Theta, theta_g: Theta) -> np.ndarray:
        f = SuperPoissonian.pmf(counts, theta_f)
        g = SuperPoissonian.pmf(counts, theta_g)
        return np.log(np.clip(f, 1e-300, None)) - np.log(np.clip(g, 1e-300, None))

    def llr(self, frames: np.ndarray) -> np.ndarray:
        """(n_frames, n_sites) log-likelihood ratios."""
        counts = self.roi.counts(frames)
        out = np.empty_like(counts)
        for k, m in enumerate(self.models_):
            out[:, k] = self._llr(counts[:, k], m.theta_f, m.theta_g)
        return out

    def predict(self, frames: np.ndarray, use_network: bool = False) -> np.ndarray:
        """Binary per-site predictions.

        ``use_network=False`` is the independent per-site likelihood-ratio
        decision. ``use_network=True`` uses the register-coupled PI-Network.
        Both are reported in the protocol, because the gap between them is
        itself the evidence for how much of this task needs cross-site
        coupling.
        """
        if use_network:
            if self.pi_network_ is None:
                raise RuntimeError("call fit_pi_network() before use_network=True")
            return self._network_scores(frames) >= getattr(self, "pi_threshold_", 0.5)
        thr = np.array([m.threshold for m in self.models_])
        return self.llr(frames) >= thr[None, :]

    def predict_proba(self, frames: np.ndarray, use_network: bool = False) -> np.ndarray:
        """Per-site bright probability.

        Without the network this is the single-observation posterior
        ``P(bright | n_k)`` under a uniform prior: for N = 1 the mixture
        posterior of Eq. (2) collapses to ``sigmoid(LLR)``, so no extra
        calibration parameter is introduced and the probability is the
        method's own rather than a fitted wrapper.
        """
        if use_network:
            if self.pi_network_ is None:
                raise RuntimeError("call fit_pi_network() before use_network=True")
            return self._network_scores(frames)
        s = self.llr(frames)
        return 1.0 / (1.0 + np.exp(-np.clip(s, -700, 700)))

    def mean_overlap(self) -> float:
        """Mean Bhattacharyya overlap O(f,g), the paper's difficulty measure."""
        return float(np.mean([m.overlap for m in self.models_]))

    # ------------------------------------------------------------------
    # Optional register-coupled variant (SitePINetwork)
    # ------------------------------------------------------------------

    def site_features(self, frames: np.ndarray) -> np.ndarray:
        """(n_frames, n_sites, 5) inputs for :class:`SitePINetwork`.

        Per site: the log-likelihood ratio plus that site's calibrated
        ``(alpha_g, beta_g, alpha_f, beta_f)``. Deliberately nothing else --
        the network sees exactly the quantities the weakly anchored Bayesian
        calibration already produces, so the comparison isolates the effect of
        coupling sites, not the effect of feeding in more of the image.
        """
        s = self.llr(frames)
        theta = np.array(
            [[m.theta_g.alpha, m.theta_g.beta, m.theta_f.alpha, m.theta_f.beta] for m in self.models_],
            dtype=float,
        )
        theta = np.broadcast_to(theta, (frames.shape[0],) + theta.shape)
        return np.concatenate([s[..., None], theta], axis=-1).astype(np.float32)

    def _network_scores(self, frames_or_feats: np.ndarray) -> np.ndarray:
        """Accept either raw frames ``(N, H, W)`` or precomputed features.

        Both are three-dimensional, so the shape alone is ambiguous; the test
        is whether the trailing axes match ``(n_sites, n_features)``.
        """
        mu, sd = self._pi_norm
        arr = np.asarray(frames_or_feats)
        is_feats = arr.ndim == 3 and arr.shape[1:] == (self.n_sites, mu.shape[0])
        feats = arr if is_feats else self.site_features(arr)
        return self.pi_network_.predict_proba((feats - mu) / sd)

    def _select_network_threshold(self, feats: np.ndarray, labels: np.ndarray) -> float:
        p = self._network_scores(feats)
        grid = np.linspace(0.05, 0.95, 91)
        return float(grid[int(np.argmax([classification_fidelity(labels, p >= t) for t in grid]))])

    def fit_pi_network(
        self,
        train_frames: np.ndarray,
        train_labels: np.ndarray,
        val_frames: np.ndarray,
        val_labels: np.ndarray,
        epochs: int = 30,
        batch_size: int = 32,
        lr: float = 1e-3,
        hidden: int = 128,
        n_blocks: int = 2,
        device: str = "cpu",
        seed: int = 0,
        verbose: bool = True,
    ) -> "SiteBayesianReadout":
        """Train the site-equivariant PI-Network on top of the calibrated LLRs.

        Model selection is by validation classification fidelity and the best
        checkpoint is restored at the end, so the reported test number comes
        from a state chosen without ever touching the test split.
        """
        import torch
        from torch.utils.data import DataLoader, TensorDataset

        from .pi_network import SitePINetwork

        torch.manual_seed(seed)
        Xtr = torch.as_tensor(self.site_features(train_frames))
        Ytr = torch.as_tensor(train_labels.astype(np.float32))
        Xva = self.site_features(val_frames)

        # Feature standardisation from the training split only.
        flat = Xtr.reshape(-1, Xtr.shape[-1])
        mu, sd = flat.mean(0), flat.std(0).clamp_min(1e-6)
        self._pi_norm = (mu.numpy(), sd.numpy())
        Xtr = (Xtr - mu) / sd

        net = SitePINetwork(n_features=Xtr.shape[-1], hidden=hidden, n_blocks=n_blocks).to(device)
        opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
        loss_fn = torch.nn.BCEWithLogitsLoss()
        loader = DataLoader(TensorDataset(Xtr, Ytr), batch_size=batch_size, shuffle=True)

        self.pi_network_ = net
        best_fid, best_state, last_loss = -np.inf, None, float("nan")
        for ep in range(1, epochs + 1):
            net.train()
            for xb, yb in loader:
                xb, yb = xb.to(device), yb.to(device)
                loss = loss_fn(net(xb), yb)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                last_loss = float(loss.item())
            sched.step()

            fid = classification_fidelity(val_labels, self._network_scores(Xva) >= 0.5)
            if fid > best_fid:
                best_fid = fid
                best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
            if verbose and (ep % 5 == 0 or ep == 1):
                print(f"  PI-Net epoch {ep:3d}/{epochs}  loss={last_loss:.4f}  val_fidelity={fid:.5f}")

        if best_state is not None:
            net.load_state_dict(best_state)
        self.pi_network_ = net
        self.pi_threshold_ = self._select_network_threshold(Xva, val_labels)
        if verbose:
            print(f"  PI-Net best val_fidelity={best_fid:.5f}, threshold={self.pi_threshold_:.3f}")
        return self


class EnsembleReadout:
    """Mode A: the paper's own task, unchanged.

    Given a block of frames all prepared in the same state, recover the
    occupation ``l`` per site and its posterior uncertainty ``Delta l``, and
    score with the paper's relative fidelity, Eq. (3).
    """

    def __init__(self, roi: ROIExtractor, n_grid: int = 501, max_iter: int = 10):
        self.roi = roi
        self.n_grid = n_grid
        self.max_iter = max_iter

    def anchor_dark(self, dark_frames: np.ndarray) -> list:
        counts = self.roi.counts(dark_frames)
        return [SuperPoissonian.fit_moments(counts[:, k]) for k in range(counts.shape[1])]

    def infer(self, frames: np.ndarray, theta_g_per_site: list) -> tuple:
        """Return ``(l_mean, l_std)``, each of shape ``(n_sites,)``."""
        counts = self.roi.counts(frames)
        n_sites = counts.shape[1]
        l_mean = np.empty(n_sites)
        l_std = np.empty(n_sites)
        for k in range(n_sites):
            res = weakly_anchored_em(
                counts[:, k], theta_g_per_site[k], n_grid=self.n_grid, max_iter=self.max_iter
            )
            l_mean[k], l_std[k] = res.l_mean, res.l_std
        return l_mean, l_std

    def infer_with_network(self, frames: np.ndarray, theta_g_per_site: list, network) -> tuple:
        """Same estimand, but the posterior comes from the PI-Network.

        Used to reproduce the paper's speedup claim: the EM loop is replaced by
        a single forward pass, and the two posteriors should agree.
        """
        counts = self.roi.counts(frames)
        n_shots, n_sites = counts.shape
        l_grid = np.linspace(0.0, 1.0, network.config.n_grid)
        dl = float(l_grid[1] - l_grid[0])

        l_mean = np.empty(n_sites)
        l_std = np.empty(n_sites)
        for k in range(n_sites):
            theta_g = theta_g_per_site[k]
            theta_f = weakly_anchored_em(
                counts[:, k], theta_g, n_grid=self.n_grid, max_iter=self.max_iter
            ).theta_f
            s = SiteBayesianReadout._llr(counts[:, k], theta_f, theta_g)
            aux = np.array(
                [np.log(n_shots), theta_g.alpha, theta_g.beta, theta_f.alpha, theta_f.beta]
            )
            post = network.posterior(s, aux)[0]
            p = post * dl
            p = p / max(p.sum(), 1e-300)
            l_mean[k] = float(np.sum(p * l_grid))
            l_std[k] = float(np.sqrt(max(np.sum(p * (l_grid - l_mean[k]) ** 2), 0.0)))
        return l_mean, l_std
