"""Twelve-group relation-composition task: interaction recovery and selection.

Source Fig. 3, Tables 7, 8, 10-12 (App. I.4, I.6, J.3-J.5). At new-observation
budget m the fitted coalitions are the 17 gate coalitions that are not of size
3 plus the first m of ``configs/new_observed_masks.txt`` (``fit_d12.py``); the
source split them 80/20 into fit and validation, the GP uses them all. All 220
size-3 coalitions are held out. Metrics follow ``analyze_d12.py``, whose
average-precision and third-difference definitions are copied here: AP of the
complete triples ranked by |predicted third difference at the empty
coalition|, top-1 and top-8 completeness of size-3 selection, and Shapley- and
Banzhaf-based selection of three groups.
"""
from __future__ import annotations

import csv
import itertools

import numpy as np
from scipy.stats import spearmanr

from .. import methods
from . import TNDM

D, Q = 12, 10
SEEDS = (43, 59)            # the source's blind seeds (seed 31, its development seed, lacks the gate cells)
BUDGETS = (32, 48, 64, 96, 128, 160)
HERE = TNDM / "experiments/reasoning/d12"
REQUIRED_MASKS = (4095, 21, 7, 4092, 4083, 4047, *tuple(1 << i for i in range(12)), 0)
QUERY_BLOCKS = (
    (0, 1, 2, 3, 4, 5), (6, 7, 8, 9, 10, 11),
    (0, 2, 4, 6, 8, 10), (1, 3, 5, 7, 9, 11),
    (0, 1, 4, 5, 8, 9), (2, 3, 6, 7, 10, 11),
    (0, 3, 4, 7, 8, 11), (1, 2, 5, 6, 9, 10),
    (0, 2, 5, 7, 9, 11), (1, 3, 4, 6, 8, 10),
)   # = experiments/reasoning/common/transformer_query_paths_d12.py
PROVIDERS = tuple(tuple((b[i], b[i + 1]) for i in range(0, 6, 2)) for b in QUERY_BLOCKS)
SPLITS = ("target", "interaction_audit", "gate", "observed", "loo")


def completed(mask: int, query: int) -> bool:
    return all(any(mask & (1 << g) for g in pair) for pair in PROVIDERS[query])


def average_precision(labels: np.ndarray, score: np.ndarray) -> float:
    order = np.lexsort((np.arange(len(score)), -score))
    ranked = labels[order].astype(float)
    if not ranked.sum():
        return float("nan")
    precision = np.cumsum(ranked) / np.arange(1, len(ranked) + 1)
    return float(np.sum(precision * ranked) / ranked.sum())


def delta3(values: np.ndarray, mask: int) -> float:
    members = [i for i in range(D) if mask & (1 << i)]
    singles = [1 << i for i in members]
    pairs = [mask ^ s for s in singles]
    return float(values[mask] - sum(values[p] for p in pairs) + sum(values[s] for s in singles) - values[0])


def shapley_banzhaf(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    shapley, banzhaf = np.zeros(D), np.zeros(D)
    every = np.arange(1 << D)
    size = np.asarray([int(m).bit_count() for m in every])
    for g in range(D):
        absent = every[(every & (1 << g)) == 0]
        marginal = values[absent | (1 << g)] - values[absent]
        banzhaf[g] = marginal.mean()
        shapley[g] = np.mean([marginal[size[absent] == s].mean() for s in range(D)])
    return shapley, banzhaf


_CELLS: dict = {}


def cells(seed: int) -> dict[int, np.ndarray]:
    """mask -> (10,) log probabilities, splits searched in the source's priority order."""
    if seed not in _CELLS:
        table: dict = {}
        with (HERE / "cubes" / f"d12_cells_seed{seed}.csv").open(newline="") as handle:
            for row in csv.DictReader(handle):
                if row["split"] in SPLITS:
                    cell = table.setdefault((row["split"], int(row["mask"])), [None] * Q)
                    cell[int(row["query"])] = float(row["log_probability"])
        out = {}
        for (split, mask), value in table.items():
            if mask not in out or SPLITS.index(split) < SPLITS.index(out[mask][0]):
                out[mask] = (split, value)
        _CELLS[seed] = {m: np.asarray(v, dtype=float) for m, (_, v) in out.items()}
    return _CELLS[seed]


def observed_masks(budget: int) -> np.ndarray:
    gate = [m for m in REQUIRED_MASKS if m.bit_count() != 3]
    new = [int(x) for x in (HERE / "configs/new_observed_masks.txt").read_text().split()]
    return np.asarray(gate + new[:budget], dtype=np.int64)


def units(shard: str = "all") -> list[str]:
    seeds = SEEDS if shard == "all" else (int(shard.removeprefix("seed")),)
    return [f"s{s}-m{b}" for s in seeds for b in BUDGETS]


def evaluate(prediction: np.ndarray, seed: int) -> dict:
    """``prediction`` is the (4096, 10) predicted table."""
    actual_cells = cells(seed)
    target = np.asarray([m for m in range(1 << D) if m.bit_count() == 3])
    faces = [0] + [m for m in range(1 << D) if m.bit_count() in (1, 2)] + target.tolist()
    actual = np.full((1 << D, Q), np.nan)
    for m in faces:
        actual[m] = actual_cells[m]
    per_query = []
    for q in range(Q):
        y, p = actual[target, q], prediction[target, q]
        complete = np.asarray([completed(int(m), q) for m in target])
        order = np.lexsort((target, -p))
        top1, top8 = int(target[order[0]]), target[order[:8]]
        true_d = np.asarray([delta3(actual[:, q], int(m)) for m in target])
        pred_d = np.asarray([delta3(prediction[:, q], int(m)) for m in target])
        sh, bz = shapley_banzhaf(prediction[:, q])
        picks = {}
        for name, scores in (("shapley", sh), ("banzhaf", bz)):
            mask = int(sum(1 << int(g) for g in np.lexsort((np.arange(D), -scores))[:3]))
            picks[name] = completed(mask, q)
        per_query.append({
            "top1_complete": completed(top1, q), "top1_regret": float(np.max(y) - actual[top1, q]),
            "top8_complete_fraction": float(np.mean([completed(int(m), q) for m in top8])),
            "selection_ap": average_precision(complete, p),
            "complete_interaction_ap": average_precision(complete, np.abs(pred_d)),
            "delta3_mae": float(np.mean(np.abs(pred_d - true_d))),
            "delta3_spearman": float(spearmanr(pred_d, true_d).statistic) if np.std(pred_d) > 1e-12 else 0.0,
            "shapley_selection_complete": picks["shapley"], "banzhaf_selection_complete": picks["banzhaf"]})
    keys = per_query[0].keys()
    return {k: float(np.mean([r[k] for r in per_query])) for k in keys}


def run_unit(unit: str, method: str) -> dict:
    seed, budget = (int(part[1:]) for part in unit.split("-"))
    values = cells(seed)
    observed = observed_masks(budget)
    y = np.vstack([values[int(m)] for m in observed])
    model = methods.fit(method, observed, y, D)
    out = evaluate(model.predict(np.arange(1 << D)), seed)
    out.update(seed=seed, budget=budget, n_observed=len(observed), fit_seconds=model.fit_seconds,
               order=methods.order(model))
    return out
