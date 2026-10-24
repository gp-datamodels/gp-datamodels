"""Eight-group relation-composition task, exhaustive cube (source App. J.1, Table 18).

The samples are those of the source's saved 20 repeats
(``experiments/reasoning/d8/results/fits/repeat_<r>.npz``, drawn with
``default_rng(7000 + r)``), so every family is scored on identical observations.
Metrics: held-out NMSE (mean over the eight sequence-mean outputs of MSE /
variance over unobserved cells) and relative l2 errors of Shapley values and
pairwise Shapley interactions against the exact cube.
"""
from __future__ import annotations

import sys

import numpy as np

from tndm.attribution import exact

from .. import methods
from . import TNDM

D, Q = 8, 8
BUDGETS = (24, 32, 48, 64, 128)
REPEATS = tuple(range(1, 21))
FITS = TNDM / "experiments/reasoning/d8/results/fits"
SOURCE_FAMILIES = ("linear", "degree2_ridge", "degree3_ridge", "sparse_degree3", "rbf_kernel",
                   "cp", "tt", "ht", "tt_ladder")


def cube() -> np.ndarray:
    sys.path.insert(0, str(TNDM / "experiments/reasoning/d8/src"))
    import cube as source_cube  # noqa: E402
    logp, _, _ = source_cube.load_query_cube()
    return logp


def units(shard: str = "all") -> list[str]:
    reps = REPEATS if shard == "all" else (int(shard.removeprefix("r")),)
    return [f"r{r}-m{b}" for r in reps for b in BUDGETS]


def sample(repeat: int, budget: int) -> np.ndarray:
    """The source's sample, regenerated as in fit_budgets.py (one generator, budgets in order)."""
    generator = np.random.default_rng(7000 + repeat)
    for b in BUDGETS:
        drawn = np.sort(generator.choice(1 << D, b, replace=False))
        if b == budget:
            return drawn
    raise ValueError(budget)


def score(table: np.ndarray, truth: np.ndarray, observed: np.ndarray) -> dict:
    held = np.setdiff1d(np.arange(1 << D), observed)
    y = truth[held]
    nmse = ((table[held] - y) ** 2).mean(0) / y.var(0)
    rel = lambda a, b: np.linalg.norm(a - b, axis=0) / np.linalg.norm(b, axis=0)
    sh = rel(exact.shapley_values(table), exact.shapley_values(truth))
    sii = rel(exact.pair_interactions(table), exact.pair_interactions(truth))
    return {"nmse": float(nmse.mean()), "nmse_per_output": nmse.tolist(),
            "shapley_rel_l2": float(sh.mean()), "sii_rel_l2": float(sii.mean())}


def run_unit(unit: str, method: str) -> dict:
    repeat, budget = (int(part[1:]) for part in unit.split("-"))
    truth = cube()
    observed = sample(repeat, budget)
    if method == "source":
        saved = np.load(FITS / f"repeat_{repeat}.npz")
        return {fam: score(saved[f"m{budget}__{fam}"], truth, observed) for fam in SOURCE_FAMILIES}
    model = methods.fit(method, observed, truth[observed], D)
    out = score(model.predict(np.arange(1 << D)), truth, observed)
    out.update(fit_seconds=model.fit_seconds, repeat=repeat, budget=budget, order=methods.order(model))
    return out
