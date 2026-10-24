"""Gaussian-process datamodels with presence-aware product covariances.

The prior over a set function v: {0,1}^d -> R is a Gaussian process whose
covariance is a sum of R products over training groups,

    k(z, z') = sum_r exp(a_r + b_r.(z + z')/2) * prod_j exp(-g_rj [z_j != z'_j]).

Each factor is a 2x2 positive semidefinite block over the presence and absence
of group j: g_rj >= 0 is its relevance (g = 0: the group does not change the
outcome) and b_rj lets the prior variance depend on whether the group is
present, so small coalitions, the empty one in particular, can vary more than
large ones. With b = 0 and R = 1 the covariance is the stationary weighted
Hamming (ARD) kernel; with b = 0, R = 1 and all g equal it is the RBF kernel on
inclusion vectors.

Hyperparameters are fitted per output by maximizing the log marginal
likelihood plus a log hyperprior (MAP-II) on all observations, with a
hierarchical prior that shrinks the per-group log relevances of a component to
a shared level: log g_rj = log gbar_r + delta_rj, delta_rj ~ N(0, tau^2).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, replace

import numpy as np
import torch

from .coalitions import bits_of

torch.set_default_dtype(torch.float64)

__all__ = ["GPConfig", "ProductCovariance", "GPDatamodel"]


@dataclass
class GPConfig:
    rank: int = 1                 # number of product components R
    ard: bool = True              # per-group relevances (else one per component)
    amplitude: bool = True        # presence-dependent log amplitude b_r
    tau: float = 1.0              # prior sd of per-group log-relevance deviations
    amp_sd: float = 0.5           # prior sd of the presence log-amplitude slopes
    noise_loc: float = math.log(0.01)   # prior mean of the log noise variance (standardized y)
    noise_sd: float = 2.0         # prior sd of the log noise variance
    noise_floor: float = 1e-6     # added to the noise variance, standardized units
    restarts: int = 3
    max_iter: int = 200
    share: str = "none"           # "none": one kernel per output; "all": one kernel for every output
    seed: int = 0
    extra: dict = field(default_factory=dict)


class ProductCovariance(torch.nn.Module):
    """Sum of R presence-aware product covariances over d binary modes."""

    def __init__(self, d: int, rank: int = 1, ard: bool = True, amplitude: bool = True,
                 init: int = 0, seed: int = 0):
        super().__init__()
        self.d, self.rank, self.ard, self.amplitude = d, rank, ard, amplitude
        gen = torch.Generator().manual_seed(1000 * seed + init)
        # init 0: every group mildly relevant, one shared level; init 1: more local;
        # init >= 2: random deviations. Components are spread over levels.
        level = {0: 1.0 / d, 1: 4.0 / d}.get(init, 2.0 / d)
        spread = torch.linspace(-0.5, 0.5, rank) if rank > 1 else torch.zeros(1)
        self.log_gbar = torch.nn.Parameter(math.log(level) + spread)
        scale = 0.0 if init < 2 else 0.7
        if rank > 1:
            scale = max(scale, 0.5)
        self.delta = torch.nn.Parameter(scale * torch.randn(rank, d, generator=gen)) if ard else None
        self.log_scale = torch.nn.Parameter(torch.full((rank,), -math.log(rank)))
        self.b = torch.nn.Parameter(torch.zeros(rank, d)) if amplitude else None

    def gammas(self) -> torch.Tensor:
        g = self.log_gbar[:, None].expand(self.rank, self.d)
        if self.delta is not None:
            g = g + self.delta
        return torch.exp(g)

    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        g = self.gammas()
        out = torch.zeros(len(z1), len(z2))
        for r in range(self.rank):
            w1, w2 = z1 * g[r], z2 * g[r]
            # weighted Hamming distance: sum_j g_j (z1_j + z2_j - 2 z1_j z2_j)
            h = w1.sum(1)[:, None] + w2.sum(1)[None, :] - 2.0 * (w1 @ z2.T)
            log_k = self.log_scale[r] - h
            if self.b is not None:
                log_k = log_k + 0.5 * ((z1 @ self.b[r])[:, None] + (z2 @ self.b[r])[None, :])
            out = out + torch.exp(log_k)
        return out

    def diagonal(self, z: torch.Tensor) -> torch.Tensor:
        out = torch.zeros(len(z))
        for r in range(self.rank):
            log_k = self.log_scale[r] + (z @ self.b[r] if self.b is not None else 0.0)
            out = out + torch.exp(log_k)
        return out

    @torch.no_grad()
    def factors(self, z_points: torch.Tensor):
        """Product form of the kernel sections k(., z_i).

        Returns (C, U0, U1) with shapes (R, n), (R, n, d), (R, n, d) such that
        k(z, z_i) = sum_r C[r, i] prod_j (U0[r, i, j] if z_j = 0 else U1[r, i, j]).
        """
        g = self.gammas()
        b = self.b if self.b is not None else torch.zeros(self.rank, self.d)
        zi = z_points[None, :, :]                       # 1, n, d
        U0 = torch.exp(-g[:, None, :] * zi)             # z_j = 0 differs when z_ij = 1
        U1 = torch.exp(0.5 * b[:, None, :] - g[:, None, :] * (1.0 - zi))
        C = torch.exp(self.log_scale[:, None] + 0.5 * (z_points @ b.T).T)
        return C, U0, U1

    def log_prior(self, tau: float, amp_sd: float) -> torch.Tensor:
        lp = torch.zeros(())
        if self.delta is not None:
            lp = lp - 0.5 * (self.delta ** 2).sum() / tau ** 2
        if self.b is not None:
            lp = lp - 0.5 * (self.b ** 2).sum() / amp_sd ** 2
        return lp


class _OutputGP(torch.nn.Module):
    """Covariance and noise shared by the columns of a standardized target block."""

    def __init__(self, d: int, cfg: GPConfig, init: int):
        super().__init__()
        self.cfg = cfg
        self.cov = ProductCovariance(d, cfg.rank, cfg.ard, cfg.amplitude, init=init, seed=cfg.seed)
        self.log_noise = torch.nn.Parameter(torch.tensor(cfg.noise_loc + (0.0 if init % 2 == 0 else 2.0)))

    def noise(self) -> torch.Tensor:
        return torch.exp(self.log_noise) + self.cfg.noise_floor

    def objective(self, z: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        """Negative log marginal likelihood plus negative log hyperprior, per observation."""
        n, q = Y.shape
        K = self.cov(z, z) + self.noise() * torch.eye(n)
        L = torch.linalg.cholesky(K)
        A = torch.cholesky_solve(Y, L)
        nll = 0.5 * (Y * A).sum() + q * torch.log(torch.diagonal(L)).sum()
        lp = self.cov.log_prior(self.cfg.tau, self.cfg.amp_sd)
        lp = lp - 0.5 * ((self.log_noise - self.cfg.noise_loc) / self.cfg.noise_sd) ** 2
        return (nll / q - lp) / n


def _optimize(model: _OutputGP, z: torch.Tensor, Y: torch.Tensor, max_iter: int) -> float:
    """L-BFGS from the model's current state; leaves the model at its best iterate."""
    opt = torch.optim.LBFGS(model.parameters(), lr=1.0, max_iter=max_iter,
                            tolerance_grad=1e-7, tolerance_change=1e-10,
                            line_search_fn="strong_wolfe")
    state = {"loss": math.inf, "params": None}

    def closure():
        opt.zero_grad()
        try:
            loss = model.objective(z, Y)
        except RuntimeError:          # Cholesky failure: an infinite loss rejects the step
            return torch.tensor(math.inf, requires_grad=True)
        if not torch.isfinite(loss):
            return torch.tensor(math.inf, requires_grad=True)
        loss.backward()
        if float(loss) < state["loss"]:
            state["loss"] = float(loss)
            state["params"] = {k: v.detach().clone() for k, v in model.state_dict().items()}
        return loss

    try:
        opt.step(closure)
    except RuntimeError:
        pass
    if state["params"] is None:
        return math.inf
    model.load_state_dict(state["params"])
    return state["loss"]


def _restarts(z, Y, cfg: GPConfig):
    best = None
    for init in range(cfg.restarts):
        model = _OutputGP(z.shape[1], cfg, init)
        loss = _optimize(model, z, Y, cfg.max_iter)
        if math.isfinite(loss) and (best is None or loss < best[0]):
            best = (loss, model, init)
    if best is None:
        raise RuntimeError("every restart failed")
    return best


@torch.no_grad()
def _grow(previous: _OutputGP, cfg: GPConfig, init: int) -> _OutputGP:
    """A model of configuration cfg whose first components copy ``previous``.

    Added amplitude slopes start at zero and added components at a weight of
    e^-8 relative to the first, so the new model starts where the previous one
    stopped and its MAP objective can only improve on it.
    """
    model = _OutputGP(previous.cov.d, cfg, init)
    old, new = previous.cov, model.cov
    k = old.rank
    new.log_gbar[:k] = old.log_gbar
    new.log_scale[:k] = old.log_scale
    new.log_scale[k:] = old.log_scale.max() - 8.0
    if new.delta is not None and old.delta is not None:
        new.delta[:k] = old.delta
    if new.b is not None:
        new.b.zero_()
        if old.b is not None:
            new.b[:k] = old.b
    model.log_noise.copy_(previous.log_noise)
    return model


def n_hyperparameters(cfg: GPConfig, d: int) -> int:
    """Free hyperparameters of a covariance with cfg.rank components, noise included."""
    per_component = 2 + (d if cfg.ard else 0) + (d if cfg.amplitude else 0)
    return cfg.rank * per_component + 1


def _best(candidates):
    finite = [c for c in candidates if math.isfinite(c[0])]
    if not finite:
        raise RuntimeError("every fit failed")
    return min(finite, key=lambda c: c[0])


def _grown_candidates(previous: _OutputGP, stage: GPConfig, z, Y):
    """Continuation from ``previous``: the added component starts dense (trial 0) or perturbed."""
    out = []
    for trial in range(2):
        grown = _grow(previous, stage, init=2 + trial)
        if trial == 0 and grown.cov.delta is not None:
            with torch.no_grad():
                grown.cov.delta[stage.rank - 1].zero_()
        out.append((_optimize(grown, z, Y, stage.max_iter), grown))
    return out


def _fit_block(z: torch.Tensor, Y: torch.Tensor, cfg: GPConfig):
    """MAP-II fit of one shared covariance to the standardized columns Y (n, q).

    Strategy ``select`` (the default) fits the nested models in turn. One
    component: stationary first (``restarts`` initializations), then with the
    presence-dependent amplitude, both continued from the stationary optimum and
    from ``restarts`` fresh initializations. Each further component: continued
    from the previous model and from fresh joint initializations, the best MAP
    objective kept. Components are added while the Bayesian information
    criterion, n J + (k/2) log n with J the MAP objective per observation and k
    the number of hyperparameters, decreases, up to ``cfg.rank``. Strategy
    ``direct`` fits ``cfg.rank`` components from fresh initializations only.
    """
    strategy = cfg.extra.get("strategy", "select")
    if strategy == "direct":
        return _restarts(z, Y, cfg)
    n, d = z.shape

    def bic(value: float, stage: GPConfig) -> float:
        return n * value + 0.5 * n_hyperparameters(stage, d) * math.log(n)

    stationary = replace(cfg, rank=1, amplitude=False)
    loss, model, init = _restarts(z, Y, stationary)
    fits = [(bic(loss, stationary), loss, model)]
    previous = model
    if cfg.amplitude:
        one = replace(cfg, rank=1)
        grown = _grow(model, one, init)
        candidates = [(_optimize(grown, z, Y, cfg.max_iter), grown)]
        fresh_loss, fresh_model, _ = _restarts(z, Y, one)
        candidates.append((fresh_loss, fresh_model))
        loss, previous = _best(candidates)
        fits.append((bic(loss, one), loss, previous))
    score = min(f[0] for f in fits)
    for rank in range(2, cfg.rank + 1):
        stage = replace(cfg, rank=rank)
        candidates = _grown_candidates(previous, stage, z, Y)
        fresh_loss, fresh_model, _ = _restarts(z, Y, stage)
        candidates.append((fresh_loss, fresh_model))
        loss, previous = _best(candidates)
        fits.append((bic(loss, stage), loss, previous))
        if fits[-1][0] >= score:
            break
        score = fits[-1][0]
    _, loss, model = min(fits, key=lambda f: f[0])
    return loss, model, init


class GPDatamodel:
    """Fit on observed coalitions, then predict any coalition and answer attribution queries."""

    def __init__(self, d: int, config: GPConfig | None = None):
        self.d = d
        self.cfg = config or GPConfig()
        self.blocks: list[dict] = []

    # -- fitting --------------------------------------------------------------
    def fit(self, masks, y) -> "GPDatamodel":
        z = torch.as_tensor(bits_of(masks, self.d))
        y = np.asarray(y, dtype=np.float64)
        if y.ndim == 1:
            y = y[:, None]
        self.n_outputs = y.shape[1]
        self.mu = y.mean(0)
        self.sd = y.std(0)
        self.sd[self.sd < 1e-12] = 1.0
        Ys = torch.as_tensor((y - self.mu) / self.sd)
        groups = ([list(range(self.n_outputs))] if self.cfg.share == "all"
                  else [[q] for q in range(self.n_outputs)])
        self.blocks = []
        for cols in groups:
            loss, model, init = _fit_block(z, Ys[:, cols], self.cfg)
            with torch.no_grad():
                K = model.cov(z, z) + model.noise() * torch.eye(len(z))
                L = torch.linalg.cholesky(K)
                alpha = torch.cholesky_solve(Ys[:, cols], L)
            self.blocks.append({"cols": cols, "model": model, "L": L, "alpha": alpha,
                                "Y": Ys[:, cols], "objective": loss, "init": init})
        self.z = z
        return self

    # -- prediction -------------------------------------------------------------
    @torch.no_grad()
    def predict(self, masks, chunk: int = 4096) -> np.ndarray:
        zs = torch.as_tensor(bits_of(masks, self.d))
        out = np.empty((len(zs), self.n_outputs))
        for start in range(0, len(zs), chunk):
            part = zs[start:start + chunk]
            for blk in self.blocks:
                ks = blk["model"].cov(part, self.z)
                out[start:start + chunk, blk["cols"]] = (ks @ blk["alpha"]).numpy()
        return out * self.sd + self.mu

    @torch.no_grad()
    def predict_var(self, masks, include_noise: bool = False, chunk: int = 2048) -> np.ndarray:
        zs = torch.as_tensor(bits_of(masks, self.d))
        out = np.empty((len(zs), self.n_outputs))
        for start in range(0, len(zs), chunk):
            part = zs[start:start + chunk]
            for blk in self.blocks:
                m = blk["model"]
                ks = m.cov(part, self.z)
                v = torch.cholesky_solve(ks.T, blk["L"])
                var = m.cov.diagonal(part) - (ks * v.T).sum(1)
                if include_noise:
                    var = var + m.noise()
                out[start:start + chunk, blk["cols"]] = var.clamp_min(0.0).numpy()[:, None]
        return out * self.sd ** 2

    @torch.no_grad()
    def posterior_cov(self, masks, output: int = 0) -> np.ndarray:
        """Posterior covariance of the noiseless function on a set of coalitions."""
        zs = torch.as_tensor(bits_of(masks, self.d))
        blk = next(b for b in self.blocks if output in b["cols"])
        m = blk["model"]
        ks = m.cov(zs, self.z)
        v = torch.cholesky_solve(ks.T, blk["L"])
        cov = m.cov(zs, zs) - ks @ v
        return cov.numpy() * self.sd[output] ** 2

    @torch.no_grad()
    def loo(self) -> tuple[np.ndarray, np.ndarray]:
        """Closed-form leave-one-out predictive means and variances (Rasmussen and Williams, eq. 5.12)."""
        means = np.empty((len(self.z), self.n_outputs))
        variances = np.empty_like(means)
        for blk in self.blocks:
            dg = torch.diagonal(torch.cholesky_inverse(blk["L"]))[:, None]
            means[:, blk["cols"]] = (blk["Y"] - blk["alpha"] / dg).numpy()
            variances[:, blk["cols"]] = (1.0 / dg).expand(-1, len(blk["cols"])).numpy()
        return means * self.sd + self.mu, variances * self.sd ** 2

    # -- introspection ------------------------------------------------------------
    def group_relevance(self, output: int = 0) -> np.ndarray:
        """Prior variance of the part of the function that depends on each group.

        For a stationary component the Walsh terms containing group j carry
        prior variance e^{a_r} (1 - e^{-g_rj}) / 2; with an amplitude the
        component's scale is averaged over uniformly random coalitions,
        E[e^{a_r + b_r.z}] = e^{a_r} prod_j (1 + e^{b_rj}) / 2. Summed over
        components and scaled to the output's units.
        """
        blk = next(b for b in self.blocks if output in b["cols"])
        cov = blk["model"].cov
        with torch.no_grad():
            g = cov.gammas()
            weight = torch.exp(cov.log_scale)
            if cov.b is not None:
                weight = weight * torch.prod(0.5 * (1 + torch.exp(cov.b)), dim=1)
            rel = (weight[:, None] * 0.5 * (1 - torch.exp(-g))).sum(0)
        return rel.numpy() * self.sd[output] ** 2

    def hyperparameters(self, output: int = 0) -> dict:
        blk = next(b for b in self.blocks if output in b["cols"])
        m = blk["model"]
        with torch.no_grad():
            return {"relevance": m.cov.gammas().numpy(),
                    "log_scale": m.cov.log_scale.numpy(),
                    "amplitude": None if m.cov.b is None else m.cov.b.numpy(),
                    "noise": float(m.noise()) * self.sd[output] ** 2,
                    "objective": blk["objective"], "init": blk["init"]}
