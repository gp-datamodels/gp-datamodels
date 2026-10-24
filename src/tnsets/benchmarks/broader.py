"""Broader settings: AG News topics and UCI HAR (d = 10), coin-flip ICL (d = 8).

Source Table 15 (App. J.7): for random permutations of the cube, reveal a
prefix of m = 2^d/8, 2^d/4, 2^d/2 coalitions, test on the rest, report median
NMSE. The source's split seeds are not released, so the splits here are
``default_rng(split)`` for split = 0..9 and the source's fitters are rerun on
them (method "source"): its ALS families with their own validation and refit
(``experiments/attribution/lib/surrogates.py``) and its gradient-fitted TNDMs
(``tndm.fit_datamodel``, which adds the cardinality adjustment). The empty
coalition dominates these cubes' variance, so NMSE is also reported with it
removed from the test set.
"""
from __future__ import annotations

import sys

import numpy as np

from .. import methods
from . import TNDM

ATTR = TNDM / "experiments/attribution"
if str(ATTR) not in sys.path:
    sys.path.insert(0, str(ATTR))
if str(TNDM / "data") not in sys.path:
    sys.path.insert(0, str(TNDM / "data"))

CUBES = ("atlas_ag_news_topics_d10", "atlas_uci_har_clients_d10", "atlas_qwen_coin_icl_d8")
N_SPLITS = 10
ALS_FAMILIES = ("linear", "degree2_ridge", "degree3_ridge", "degree3_omp", "rbf", "cp", "tt",
                "tt_ladder", "ht_ladder")


def table(name: str) -> np.ndarray:
    from cubes import load  # data/cubes.py of the release
    cube = load(name)
    return np.asarray(cube.table(cube.output_names[0]), dtype=float), cube.d


def units(shard: str) -> list[str]:
    _, d = table(shard)
    return [f"m{(1 << d) // k}-s{s}" for k in (8, 4, 2) for s in range(N_SPLITS)]


def _scores(pred: np.ndarray, y: np.ndarray, test: np.ndarray) -> dict:
    out = {"nmse": float(np.mean((pred[test] - y[test]) ** 2) / np.var(y[test]))}
    t = test[test != 0]
    out["nmse_no_empty"] = float(np.mean((pred[t] - y[t]) ** 2) / np.var(y[t]))
    return out


def run_unit(unit: str, method: str, shard: str) -> dict:
    m, split = (int(part[1:]) for part in unit.split("-"))
    y, d = table(shard)
    perm = np.random.default_rng(split).permutation(1 << d)
    observed, test = np.sort(perm[:m]), perm[m:]
    info = {"m": m, "split": split, "empty_in_test": bool(0 in set(test.tolist()))}
    if method == "source":
        from lib import surrogates
        from tndm import fit_datamodel
        out = {}
        for fam in ALS_FAMILIES:
            fit = surrogates.FAMILIES[fam](observed, y[observed, None], d, 1234 + split)
            out["als_" + fam] = _scores(fit.prediction[:, 0], y, test)
        for fam in ("tt", "cp"):
            model, _ = fit_datamodel(observed, y[observed], d, family=fam, split_seed=split)
            out["grad_" + fam] = _scores(model.predict(np.arange(1 << d))[:, 0], y, test)
        return {**info, "families": out}
    model = methods.fit(method, observed, y[observed], d)
    out = _scores(model.predict(np.arange(1 << d))[:, 0], y, test)
    return {**info, **out, "fit_seconds": model.fit_seconds, "order": methods.order(model)}
