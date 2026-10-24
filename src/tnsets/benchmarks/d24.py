"""24-group relation-composition task (source Table 1, Tables 5, 6 and 9; App. I.4, I.6, J.1, J.2).

Training masks are the nested prefixes of ``gpu_training_masks.txt`` (256, 512,
1,024). The source fitted on 80% of each prefix and validated on 20%; the GP
uses the whole prefix, the same observed outcomes. Test NMSE is the mean over
the ten sequence-mean outputs of MSE / variance on the 256 test masks; the
effects are predicted minus observed changes from the full coalition for the
24 leave-one-out and 20 leave-role-out masks, scored on all ten outputs.
"""
from __future__ import annotations

import itertools

import numpy as np

from .. import methods
from . import TNDM

D, Q = 24, 10
SEEDS = (31, 43, 59)
BUDGETS = (256, 512, 1024)
FULL = (1 << D) - 1
HERE = TNDM / "experiments/reasoning/d24"
PROVIDERS = (
    ((3, 15, 16), (10, 12, 18)), ((6, 7, 14), (1, 19, 22)), ((4, 13, 23), (2, 8, 9)),
    ((0, 17, 21), (5, 11, 20)), ((0, 2, 9), (4, 14, 23)), ((6, 11, 21), (3, 13, 17)),
    ((10, 20, 22), (1, 5, 16)), ((7, 15, 19), (8, 12, 18)), ((12, 16, 17), (0, 1, 22)),
    ((2, 6, 10), (13, 18, 19)),
)   # = experiments/reasoning/d24/gpu/structure.py PROVIDERS


def read(name: str) -> list[int]:
    return [int(x) for x in (HERE / "configs" / name).read_text().split()]


_CUBES: dict = {}


def cube(seed: int) -> dict[int, np.ndarray]:
    if seed not in _CUBES:
        data = np.load(HERE / "gpu/cube_d24_k2_g3.npz")
        _CUBES[seed] = {int(m): row for m, row in zip(data[f"seed{seed}_masks"],
                                                      data[f"seed{seed}_log_probability"])}
    return _CUBES[seed]


def completed(mask: int, query: int) -> bool:
    return all(any(mask & (1 << g) for g in role) for role in PROVIDERS[query])


def units(shard: str = "all") -> list[str]:
    seeds = SEEDS if shard == "all" else (int(shard.removeprefix("seed")),)
    return [f"s{s}-m{b}" for s in seeds for b in BUDGETS]


def evaluate(predict, seed: int) -> dict:
    """Metrics of any predictor ``predict(masks) -> (n, 10)`` against the cube of one seed."""
    values = cube(seed)
    test = np.asarray(read("gpu_test_masks.txt"), dtype=np.int64)
    y = np.vstack([values[int(m)] for m in test])
    var = y.var(0)
    var[var < 1e-12] = 1.0
    p = predict(test)
    nmse_per_output = ((p - y) ** 2).mean(0) / var
    loo = [FULL & ~(1 << g) for g in range(D)]
    lro = [FULL & ~sum(1 << g for g in PROVIDERS[q][r]) for q in range(Q) for r in range(2)]
    masks = np.asarray([FULL] + loo + lro, dtype=np.int64)
    pe = predict(masks)
    full_obs = values[FULL]
    obs_loo = np.vstack([full_obs - values[m] for m in loo])
    obs_lro = np.vstack([full_obs - values[m] for m in lro])
    p_loo = pe[0] - pe[1:1 + D]
    p_lro = pe[0] - pe[1 + D:]
    size3 = np.asarray(sorted(sum(1 << g for g in c) for c in itertools.combinations(range(D), 3)),
                       dtype=np.int64)
    p3 = predict(size3)
    top3 = [[int(size3[i]) for i in np.lexsort((size3, -p3[:, q]))[:3]] for q in range(Q)]
    completeness = float(np.mean([[completed(m, q) for m in top3[q]] for q in range(Q)]))
    return {"nmse": float(nmse_per_output.mean()), "nmse_per_output": nmse_per_output.tolist(),
            "loo_rmse": float(np.sqrt(np.mean((p_loo - obs_loo) ** 2))),
            "lro_rmse": float(np.sqrt(np.mean((p_lro - obs_lro) ** 2))),
            "top3_size3_completeness": completeness}


def refit_source(train: np.ndarray, y: np.ndarray, seed: int) -> dict:
    """The source's families refitted with its released fitter, timed on this machine.

    Same split seed (6001) and tensor orders as its d = 24 pipeline
    (pilot/run_arm.py: role-contiguous and one shuffled order for TT, the role
    order for CP), so the predictions should match its frozen ones.
    """
    import time
    from tndm import fit_datamodel
    orders = {"tt": (role_order(), tuple(np.random.default_rng(99).permutation(D).tolist())),
              "cp": (role_order(),)}
    out = {}
    for family in ("linear", "degree2", "rbf", "cp", "tt"):
        start = time.perf_counter()
        model, _ = fit_datamodel(train, y, D, family=family, split_seed=6001,
                                 **({"orders": orders[family]} if family in orders else {}))
        seconds = time.perf_counter() - start
        out[family] = {**evaluate(model.predict, seed), "fit_seconds": seconds}
    return out


def role_order() -> tuple[int, ...]:
    """Groups ordered so that each role's providers are contiguous (pilot/game.py role_ordering)."""
    seen, order = set(), []
    for roles in PROVIDERS:
        for role in roles:
            for g in role:
                if g not in seen:
                    seen.add(g)
                    order.append(g)
    order.extend(g for g in range(D) if g not in seen)
    return tuple(order)


def run_unit(unit: str, method: str) -> dict:
    seed, budget = (int(part[1:]) for part in unit.split("-"))
    values = cube(seed)
    train = np.asarray(read("gpu_training_masks.txt")[:budget], dtype=np.int64)
    y = np.vstack([values[int(m)] for m in train])
    if method == "source":
        return {"families": refit_source(train, y, seed), "seed": seed, "budget": budget}
    model = methods.fit(method, train, y, D)
    out = evaluate(model.predict, seed)
    # structure recovery: share of each output's six support groups among its six most relevant
    hits = []
    for q in range(Q):
        relevance = model.group_relevance(q)
        top = set(np.argsort(-relevance)[:6].tolist())
        support = {g for role in PROVIDERS[q] for g in role}
        hits.append(len(top & support) / 6)
    out.update(support_recovery=float(np.mean(hits)), fit_seconds=model.fit_seconds,
               seed=seed, budget=budget, n_observed=len(train),
               components=[b["model"].cov.rank for b in model.blocks],
               amplitude=[b["model"].cov.b is not None for b in model.blocks])
    return out


def frozen_source(seed: int, budget: int) -> dict:
    """The source's frozen predictions for every family, evaluated with :func:`evaluate`."""
    f = np.load(HERE / f"gpu/fits/seed{seed}/budget_{budget}/frozen_predictions.npz")
    position = {int(m): i for i, m in enumerate(f["evaluation_masks"])}
    out = {}
    for key in f.files:
        if not key.endswith("__evaluation"):
            continue
        table = f[key]
        size3 = f["size3_masks"]
        size3_pos = {int(m): i for i, m in enumerate(size3)}
        p3 = f[key.replace("__evaluation", "__size3")]

        def predict(masks, table=table, p3=p3):
            rows = []
            for m in masks:
                m = int(m)
                rows.append(table[position[m]] if m in position else p3[size3_pos[m]])
            return np.vstack(rows)

        out[key[: -len("__evaluation")]] = evaluate(predict, seed)
    return out
