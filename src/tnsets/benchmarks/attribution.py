"""One fit, many queries: modular retraining and AG News RoBERTa-LoRA cubes, d = 8.

Source Fig. 4 and Tables 3, 14 (App. I.7, J.6). Designs, budgets (8 to 128)
and the 20 repetitions are regenerated from the source's hashed seeds
(``experiments/attribution/lib/protocol.py``, design "uniform"), and every row
is scored by its own ``runner.score_surrogate``, so the metrics are directly
comparable with its saved per-repetition results.
"""
from __future__ import annotations

import sys

import numpy as np

from .. import methods
from . import TNDM

ATTR = TNDM / "experiments/attribution"
if str(ATTR) not in sys.path:
    sys.path.insert(0, str(ATTR))
from lib import cubes, design, protocol, runner  # noqa: E402

CUBES = (["modular_d8_seed31", "modular_d8_seed43", "modular_d8_seed59"]
         + [f"agnews_lora_r{r}_seed{s}" for r in (1, 2, 4, 8, 16, 64) for s in (31, 43, 59)])
_TRUTH: dict = {}


def load(name: str):
    if name not in _TRUTH:
        cube = cubes.load(name)
        _TRUTH[name] = (cube, runner.exact_targets(cube.values, cube.d))
    return _TRUTH[name]


def units(shard: str) -> list[str]:
    cube, _ = load(shard)
    return [f"m{b}-r{r}" for b in protocol.budgets(cube.d) for r in range(protocol.REPEATS)]


def masks_for(name: str, budget: int, repeat: int) -> np.ndarray:
    cube, _ = load(name)
    rng = np.random.default_rng(protocol.seed_for(cube.name, budget, repeat, "uniform"))
    return np.asarray(design.DESIGNS["uniform"](cube.d, budget, rng))


def run_unit(unit: str, method: str, shard: str) -> dict:
    budget, repeat = (int(part[1:]) for part in unit.split("-"))
    cube, truth = load(shard)
    d = cube.d
    masks = masks_for(shard, budget, repeat)
    held = np.setdiff1d(np.arange(1 << d), masks)
    model = methods.fit(method, masks, cube.values[masks], d)
    table = model.predict(np.arange(1 << d))
    rows = {}
    for output in range(cube.n_outputs):
        row = runner.score_surrogate(table, truth, d, output, held)
        rows[cube.output_names[output]] = {k: (float(v) if v != "" else None) for k, v in row.items()}
    return {"budget": budget, "repeat": repeat, "fit_seconds": model.fit_seconds, "outputs": rows,
            "order": methods.order(model)}
