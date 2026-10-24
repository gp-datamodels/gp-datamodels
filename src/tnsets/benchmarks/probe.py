"""Attribution accuracy against fully enumerated games (source Table 17, App. J.9).

* ``qwen_probe``: the representation-probe game, seeds 31, 43, 59, with the
  source's own fit / validation / test coalition split (512 / 205 / 307) from
  ``experiments/qwen_probe/results/probe_d10_seed*.json``. The source's
  surrogates were fitted on the 512 and selected on the 205; the GP is fitted on
  the 717 observed coalitions. Its saved errors are in ``attribution_errors.csv``.
* ``uci_har``: the classifier-refit game ``atlas_uci_har_clients_d10``. The
  source's split masks are released only as hashes, so the ten splits here are
  ``default_rng(seed)``, seed = 12001..12010, with its sizes (320 fitting, 32
  validation, 64 test coalitions), and its fitters are rerun on them ("source").

Errors are RMSEs over the d groups (Shapley, Banzhaf, leave-one-out) or the
45 pairs (Shapley interaction index) against the exact game.
"""
from __future__ import annotations

import json
import sys

import numpy as np

from tndm.attribution import exact

from .. import methods
from . import TNDM

D = 10
PROBE = TNDM / "experiments" / "qwen_probe" / "results"
HAR_SEEDS = tuple(range(12001, 12011))
ATTR = TNDM / "experiments/attribution"
if str(ATTR) not in sys.path:
    sys.path.insert(0, str(ATTR))
if str(TNDM / "data") not in sys.path:
    sys.path.insert(0, str(TNDM / "data"))


def units(shard: str) -> list[str]:
    if shard == "qwen_probe":
        return [f"s{s}" for s in (31, 43, 59)]
    if shard == "uci_har":
        return [f"s{s}" for s in HAR_SEEDS]
    raise ValueError(shard)


def game(shard: str, seed: int):
    if shard == "qwen_probe":
        blob = json.loads((PROBE / f"probe_d10_seed{seed}.json").read_text())
        v = np.asarray(blob["utility"], dtype=float)
        split = blob["split"]
        return v, np.asarray(split["train_masks"]), np.asarray(split["validation_masks"]), np.asarray(split["test_masks"])
    from cubes import load
    cube = load("atlas_uci_har_clients_d10")
    v = np.asarray(cube.table(cube.output_names[0]), dtype=float)
    perm = np.random.default_rng(seed).permutation(1 << D)
    return v, np.sort(perm[:320]), np.sort(perm[320:352]), np.sort(perm[352:416])


def errors(table: np.ndarray, v: np.ndarray, test: np.ndarray) -> dict:
    rmse = lambda a, b: float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))
    return {"shapley_rmse": rmse(exact.shapley_values(table), exact.shapley_values(v)),
            "banzhaf_rmse": rmse(exact.banzhaf_values(table), exact.banzhaf_values(v)),
            "loo_rmse": rmse(exact.loo_values(table), exact.loo_values(v)),
            "sii2_rmse": rmse(exact.pair_interactions(table), exact.pair_interactions(v)),
            "test_rmse": rmse(table[test], v[test])}


def run_unit(unit: str, method: str, shard: str) -> dict:
    seed = int(unit.removeprefix("s"))
    v, train, val, test = game(shard, seed)
    observed = np.sort(np.concatenate([train, val]))
    if method == "source":
        from lib import surrogates
        from tndm import fit_datamodel
        out = {}
        for fam in ("linear", "degree2_ridge", "rbf", "cp", "tt"):
            fit = surrogates.FAMILIES[fam](observed, v[observed, None], D, seed)
            out["als_" + fam] = errors(fit.prediction[:, 0], v, test)
        for fam in ("tt", "cp"):
            model, _ = fit_datamodel(observed, v[observed], D, family=fam, split_seed=seed)
            out["grad_" + fam] = errors(model.predict(np.arange(1 << D))[:, 0], v, test)
        return {"families": out}
    model = methods.fit(method, observed, v[observed], D)
    out = errors(model.predict(np.arange(1 << D))[:, 0], v, test)
    out.update(fit_seconds=model.fit_seconds, n_observed=len(observed), order=methods.order(model))
    return out
