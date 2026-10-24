#!/usr/bin/env python3
"""Attribution without enumeration at d = 100 (Proposition 3): fit, then time the product route.

    python scripts/scaling_demo.py      # writes results/tables/scaling.json

A synthetic outcome over d = 100 groups depends on six of them, as an AND of
two ORs plus noise; 1,000 random coalitions are observed. The posterior means
of all Shapley, Banzhaf and leave-one-out values are computed by the product
formulas (2^100 coalitions could not be enumerated) and checked against the
known support.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from tnsets import methods  # noqa: E402
from tnsets.attribution import product_banzhaf, product_loo, product_shapley  # noqa: E402


def main() -> None:
    torch.set_num_threads(4)
    d, n, rng = 100, 1000, np.random.default_rng(0)
    z = rng.integers(0, 2, size=(n, d))
    support = [(3, 41, 77), (12, 58, 90)]
    y = np.ones(n)
    for role in support:
        y = y * z[:, list(role)].max(1)
    y = y + 0.02 * rng.standard_normal(n)
    masks = z                     # an inclusion matrix: integer masks cannot hold d = 100 groups
    start = time.perf_counter()
    model = methods.fit("gp-stationary", masks, y, d)
    fit_seconds = time.perf_counter() - start
    timings, values = {}, {}
    for name, fn in (("shapley", product_shapley), ("banzhaf", product_banzhaf), ("loo", product_loo)):
        start = time.perf_counter()
        values[name] = fn(model, 0)
        timings[name] = time.perf_counter() - start
    top6 = set(np.argsort(-values["shapley"])[:6].tolist())
    truth = {g for role in support for g in role}
    out = {"d": d, "n": n, "fit_seconds": fit_seconds, "seconds": timings,
           "shapley_top6_is_support": top6 == truth,
           "shapley_sum": float(values["shapley"].sum()),
           "shapley_support_mean": float(np.mean([values["shapley"][g] for g in truth])),
           "shapley_other_max_abs": float(np.max(np.abs(np.delete(values["shapley"], sorted(truth)))))}
    path = ROOT / "results" / "tables" / "scaling.json"
    path.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
