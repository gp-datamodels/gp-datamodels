"""Calibration of the posterior on fully enumerated cubes.

The exact attributions of a cube are computed from its measured outcomes, noise
included, so the posterior that should cover them is the posterior of the
realized cube: observed coalitions are known, an unobserved coalition S is
N(m(S), v(S) + s^2) with the posterior covariance of the latent function plus
the noise variance on the diagonal. Every query is linear, A v, so its
posterior is Gaussian with mean A [y_obs; m_unobs] and covariance
A_u (K_post,uu + s^2 I) A_u^T. Reported per unit: the share of exact values
inside central intervals of level 50/80/90/95%, the mean squared standardized
error (1 when calibrated), for Shapley, Banzhaf and pairwise Shapley
interactions, and the same for predictive intervals of held-out outcomes.
"""
from __future__ import annotations

import json
import sys

import numpy as np
from scipy.stats import norm

from .. import methods
from ..attribution import linear_map
from . import TNDM

LEVELS = (0.5, 0.8, 0.9, 0.95)
BUDGETS = (32, 64, 128)
REPEATS = 10
ATTR = TNDM / "experiments/attribution"
if str(ATTR) not in sys.path:
    sys.path.insert(0, str(ATTR))
if str(TNDM / "data") not in sys.path:
    sys.path.insert(0, str(TNDM / "data"))

# ":empty" variants observe the empty coalition (no training needed) in place of one random coalition
EMPTY_SHARDS = ("atlas_ag_news_topics_d10:empty", "atlas_uci_har_clients_d10:empty", "atlas_qwen_coin_icl_d8:empty")
SHARDS = (["modular_d8_seed31", "modular_d8_seed43", "modular_d8_seed59"]
          + [f"agnews_lora_r{r}_seed{s}" for r in (1, 2, 4, 8, 16, 64) for s in (31, 43, 59)]
          + ["atlas_ag_news_topics_d10", "atlas_uci_har_clients_d10", "atlas_qwen_coin_icl_d8",
             "qwen_probe_seed31", "qwen_probe_seed43", "qwen_probe_seed59", "khop_d8"])


def game(shard: str) -> tuple[np.ndarray, int]:
    """(2^d, Q) measured table and d; Q = 1 except khop_d8 (8 sequences)."""
    shard = shard.removesuffix(":empty")
    if shard.startswith(("modular", "agnews")):
        from lib import cubes
        cube = cubes.load(shard)
        return cube.values[:, [cube.primary_output]], cube.d
    if shard.startswith("atlas"):
        from cubes import load
        cube = load(shard)
        return np.asarray(cube.table(cube.output_names[0]), float)[:, None], cube.d
    if shard.startswith("qwen_probe"):
        seed = int(shard.removeprefix("qwen_probe_seed"))
        path = TNDM / "experiments" / "qwen_probe" / "results" / f"probe_d10_seed{seed}.json"
        return np.asarray(json.loads(path.read_text())["utility"], float)[:, None], 10
    if shard == "khop_d8":
        from .d8 import cube
        return cube(), 8
    raise ValueError(shard)


def units(shard: str) -> list[str]:
    return [f"m{b}-r{r}" for b in BUDGETS for r in range(REPEATS)]


def coverage(mean: np.ndarray, var: np.ndarray, truth: np.ndarray) -> dict:
    sd = np.sqrt(np.maximum(var, 1e-300))
    z = (truth - mean) / sd
    out = {f"cover{int(100 * a)}": float(np.mean(np.abs(z) <= norm.ppf(0.5 + a / 2))) for a in LEVELS}
    out["mean_z2"] = float(np.mean(z ** 2))
    out["n"] = int(len(z))
    return out


def run_unit(unit: str, method: str, shard: str) -> dict:
    budget, repeat = (int(p[1:]) for p in unit.split("-"))
    table, d = game(shard)
    n_cells = 1 << d
    base = shard.removesuffix(":empty")
    rng = np.random.default_rng([budget, repeat, sum(map(ord, base))])   # deterministic per unit
    observed = np.sort(rng.choice(n_cells, budget, replace=False))
    if shard.endswith(":empty") and 0 not in observed:
        # the empty coalition needs no training: observe it in place of one random coalition
        observed = np.sort(np.concatenate([[0], np.delete(observed, rng.integers(len(observed)))]))
    unobserved = np.setdiff1d(np.arange(n_cells), observed)
    model = methods.fit(method, observed, table[observed], d)
    every = np.arange(n_cells)
    mean_all = model.predict(every)
    out = {"budget": budget, "repeat": repeat, "fit_seconds": model.fit_seconds, "outputs": {},
           "order": methods.order(model)}
    loo_mean, loo_var = model.loo()
    for q in range(table.shape[1]):
        noise = model.hyperparameters(q)["noise"]
        cov_u = model.posterior_cov(unobserved, output=q) + noise * np.eye(len(unobserved))
        # cross-validation estimate of the covariance scale (Bachoc 2013): mean squared LOO z
        scale = float(np.mean((table[observed, q] - loo_mean[:, q]) ** 2 / loo_var[:, q]))
        realized = mean_all[:, q].copy()
        realized[observed] = table[observed, q]
        res = {"loo_scale": scale}
        pred_var = np.diag(cov_u)
        for tag, factor in (("", 1.0), ("_cv", scale)):
            res["heldout" + tag] = coverage(mean_all[unobserved, q], factor * pred_var, table[unobserved, q])
        for query in ("shapley", "banzhaf", "sii2"):
            A = linear_map(d, query)
            mean = A @ realized
            Au = A[:, unobserved]
            var = np.einsum("ij,jk,ik->i", Au, cov_u, Au)
            for tag, factor in (("", 1.0), ("_cv", scale)):
                res[query + tag] = coverage(mean, factor * var, A @ table[:, q])
        out["outputs"][str(q)] = res
    return out
