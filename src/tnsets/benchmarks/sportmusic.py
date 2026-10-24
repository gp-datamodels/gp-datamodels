"""Sport-Music, Qwen2-7B full fine-tuning, d = 10 (source Tables 2 and 13, App. J.6).

The 88 seed-0 coalitions in design order (the order of the source's
``coalition_*`` directories), the five per-class outputs, and the source's 15
shared splits (``default_rng(3)``, 30% test); the score is the per-class
held-out R^2 averaged over classes
(``experiments/sport_music/fitting_compute.py``). The released cube
``data/cubes/sportmusic_qwen2_7b_d10.npz`` replaces the per-coalition files,
of which the release ships only two.
"""
from __future__ import annotations

import importlib.util

import numpy as np

from .. import methods
from . import TNDM

D = 10
N_SPLITS = 15
SM = TNDM / "experiments/sport_music"


def _module(name: str, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def data() -> tuple[np.ndarray, np.ndarray]:
    blob = np.load(TNDM / "data/cubes/sportmusic_qwen2_7b_d10.npz")
    keep = blob["seeds"] == 0
    Z = blob["mask_bits"][keep].astype(float)
    Y = blob["values"][keep][:, 1:]
    assert len(Z) == 88 and list(blob["output_names"][1:]) == [f"class_{c}_mean_gold_logprob" for c in range(5)]
    return Z, Y


def splits() -> list[np.ndarray]:
    rng = np.random.default_rng(3)
    return [rng.permutation(88) for _ in range(N_SPLITS)]


def units(shard: str = "all") -> list[str]:
    return [f"split{i}" for i in range(N_SPLITS)]


def r2_cols(Y, P):
    out = []
    for q in range(Y.shape[1]):
        den = float(np.sum((Y[:, q] - Y[:, q].mean()) ** 2))
        out.append(np.nan if den == 0 else float(1 - np.sum((Y[:, q] - P[:, q]) ** 2) / den))
    return np.asarray(out)


def run_unit(unit: str, method: str) -> dict:
    i = int(unit.removeprefix("split"))
    Z, Y = data()
    perm = splits()[i]
    nte = max(4, int(round(0.30 * 88)))
    te, tr = perm[:nte], perm[nte:]
    masks = Z.astype(np.int64) @ (1 << np.arange(D))
    if method == "source":
        fc = _module("fitting_compute", SM / "fitting_compute.py")
        cpq = fc.CPQ
        out = {}
        for name in ("linear", "matched", "degree-2"):
            btr, bte = fc.bases(Z[tr])[name], fc.bases(Z[te])[name]
            rng = np.random.default_rng(i)
            pred = np.column_stack([fc.ridge_tuned(btr, Y[tr, c], bte, rng) for c in range(5)])
            out[name] = float(np.nanmean(r2_cols(Y[te], pred)))
        m, mu, sd = cpq.fit_cp(Z[tr], Y[tr], 5, epochs=fc.EPOCHS)
        out["cp_single_r5"] = float(np.nanmean(r2_cols(Y[te], cpq.predict(m, Z[te], mu, sd))))
        ens = np.mean([cpq.predict_fit(cpq.fit_cp(Z[tr], Y[tr], r, epochs=fc.EPOCHS, seed=s), Z[te])
                       for r in (3, 4, 5, 6) for s in (0, 1)], 0)
        out["cp_ensemble_ranks3to6"] = float(np.nanmean(r2_cols(Y[te], ens)))
        return out
    model = methods.fit(method, masks[tr], Y[tr], D)
    r2 = r2_cols(Y[te], model.predict(masks[te]))
    return {"r2": float(np.nanmean(r2)), "r2_per_class": r2.tolist(), "fit_seconds": model.fit_seconds,
            "order": methods.order(model)}
