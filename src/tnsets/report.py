"""Collect run outputs and the source's released results into tidy tables.

Run outputs are read from results/runs/ (the runs of the paper), or from the
directory named by the environment variable TNSETS_RUNS (for example out/runs/
after rerunning a campaign). Every number that reaches the paper is produced here or in
scripts/make_tables.py, from run metrics and released source artifacts; none is
typed by hand.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from .benchmarks import TNDM

ROOT = Path(__file__).resolve().parents[2]
RUNS = Path(os.environ.get("TNSETS_RUNS", ROOT / "results" / "runs"))


def _units_of(run: Path) -> dict:
    """Units of a run: in metrics.json (current layout) or units/<unit>/metrics.json (older layout)."""
    units = {}
    for path in sorted((run / "units").glob("*/metrics.json")):
        units[path.parent.name] = json.loads(path.read_text())
    summary = run / "metrics.json"
    if summary.exists():
        units.update(json.loads(summary.read_text()).get("units", {}))
    return units


def load_units(prefix: str, root: Path = RUNS) -> pd.DataFrame:
    """All unit metrics of the runs whose name starts with ``prefix``, one row per unit."""
    rows = []
    for run in sorted(root.glob(prefix + "*")):
        config_path = run / "config.json"
        if not config_path.exists():
            continue
        config = json.loads(config_path.read_text())
        for unit, metrics in _units_of(run).items():
            rows.append({"run": run.name, "method": config["method"], "shard": config["shard"],
                         "unit": unit, **metrics})
    return pd.DataFrame(rows)


# -- d = 24 --------------------------------------------------------------------------

def d24_table() -> pd.DataFrame:
    from .benchmarks import d24
    rows = []
    for seed in d24.SEEDS:
        for budget in d24.BUDGETS:
            for family, m in d24.frozen_source(seed, budget).items():
                rows.append({"method": "source:" + family, "seed": seed, "budget": budget,
                             **{k: v for k, v in m.items() if k != "nmse_per_output"}})
    ours = load_units("d24-")
    for _, r in ours.iterrows():
        rows.append({"method": r["method"], "seed": r["seed"], "budget": r["budget"],
                     **{k: r[k] for k in ("nmse", "loo_rmse", "lro_rmse", "top3_size3_completeness",
                                          "support_recovery", "fit_seconds") if k in r}})
    return pd.DataFrame(rows)


# -- d = 12 --------------------------------------------------------------------------

def d12_table() -> pd.DataFrame:
    src = pd.read_csv(TNDM / "experiments/reasoning/d12/analysis/size3_method_summary.csv")
    src = src[src.seed.isin([43, 59])].rename(columns={"new_budget": "budget"})
    src["method"] = "source:" + src["method"]
    attr = pd.read_csv(TNDM / "experiments/reasoning/d12/analysis/size3_attribution_selection.csv")
    attr = attr[attr.seed.isin([43, 59])]
    sel = (attr.groupby(["seed", "new_budget", "method", "criterion"]).complete.mean()
           .unstack("criterion").reset_index().rename(columns={"new_budget": "budget",
                                                               "shapley": "shapley_selection_complete",
                                                               "banzhaf": "banzhaf_selection_complete"}))
    sel["method"] = "source:" + sel["method"]
    src = src.merge(sel, on=["seed", "budget", "method"], how="left")
    src = src.rename(columns={"top1_complete_rate": "top1_complete"})
    ours = load_units("d12-")
    if len(ours):
        ours = ours[ours.seed.isin([43, 59])]
    return pd.concat([src, ours], ignore_index=True)


# -- d = 8 ---------------------------------------------------------------------------

def d8_table() -> pd.DataFrame:
    from .benchmarks import d8
    rows = []
    for unit in d8.units("all"):
        repeat, budget = (int(p[1:]) for p in unit.split("-"))
        for family, m in d8.run_unit(unit, "source").items():
            rows.append({"method": "source:" + family, "repeat": repeat, "budget": budget,
                         **{k: v for k, v in m.items() if k != "nmse_per_output"}})
    ours = load_units("d8-")
    for _, r in ours.iterrows():
        rows.append({"method": r["method"], "repeat": r["repeat"], "budget": r["budget"],
                     **{k: r[k] for k in ("nmse", "shapley_rel_l2", "sii_rel_l2", "fit_seconds")}})
    return pd.DataFrame(rows)


# -- modular and AG News -------------------------------------------------------------

ATTR_COLS = ["shapley_rel_l2", "banzhaf_rel_l2", "loo_rel_l2", "sii_rel_l2", "sii3_rel_l2",
             "heldout_nmse", "selection_regret_normalised"]


def attribution_table() -> pd.DataFrame:
    frames = []
    for path in sorted((TNDM / "experiments/attribution/results/question_a").glob("*.csv")):
        df = pd.read_csv(path)
        df = df[(df.method_type == "datamodel") & (df.design == "uniform")]
        df = df[["cube", "method", "output_name", "budget", "repeat"] + ATTR_COLS].copy()
        df["method"] = "source:" + df["method"]
        frames.append(df)
    ours = load_units("attribution-")
    rows = []
    for _, r in ours.iterrows():
        for output, m in r["outputs"].items():
            rows.append({"cube": r["shard"], "method": r["method"], "output_name": output,
                         "budget": r["budget"], "repeat": r["repeat"], **{k: m.get(k) for k in ATTR_COLS}})
    frames.append(pd.DataFrame(rows))
    return pd.concat(frames, ignore_index=True)


def sportmusic_table() -> pd.DataFrame:
    rows = []
    saved = json.loads((TNDM / "experiments/sport_music/results/FITTING_COMPUTE_mean_gold_logprob.json").read_text())
    for name, value in saved.items():
        rows.append({"method": "paper:" + name, "r2": value["r2"], "sd": value["sd"]})
    units = load_units("sportmusic-")
    for method, grp in units.groupby("method"):
        if method == "source":
            for col in [c for c in grp.columns if c not in ("run", "method", "shard", "unit", "unit_seconds")]:
                vals = grp[col].astype(float)
                rows.append({"method": "rerun:" + col, "r2": vals.mean(), "sd": vals.std(ddof=0)})
        else:
            rows.append({"method": method, "r2": grp["r2"].mean(), "sd": grp["r2"].std(ddof=0)})
    return pd.DataFrame(rows)


def broader_table() -> pd.DataFrame:
    units = load_units("broader-")
    rows = []
    for _, r in units.iterrows():
        base = {"cube": r["shard"], "m": r["m"], "split": r["split"], "empty_in_test": r["empty_in_test"]}
        if r["method"] == "source":
            for fam, m in r["families"].items():
                rows.append({**base, "method": "source:" + fam, **m})
        else:
            rows.append({**base, "method": r["method"], "nmse": r["nmse"], "nmse_no_empty": r["nmse_no_empty"]})
    return pd.DataFrame(rows)


def probe_table() -> pd.DataFrame:
    rows = []
    saved = pd.read_csv(TNDM / "experiments/qwen_probe/results/attribution_errors.csv")
    names = {"shapley": "shapley_rmse", "banzhaf": "banzhaf_rmse", "loo": "loo_rmse",
             "pair_shapley_interaction": "sii2_rmse"}
    for (seed, family), grp in saved.groupby(["seed", "family"]):
        row = {"shard": "qwen_probe", "seed": seed, "method": "source:" + family}
        for _, q in grp.iterrows():
            row[names[q["query"]]] = q["rmse"]
        rows.append(row)
    units = load_units("probe-")
    for _, r in units.iterrows():
        seed = int(r["unit"].removeprefix("s"))
        if r["method"] == "source":
            for fam, m in r["families"].items():
                rows.append({"shard": r["shard"], "seed": seed, "method": "source:" + fam, **m})
        else:
            rows.append({"shard": r["shard"], "seed": seed, "method": r["method"],
                         **{k: r[k] for k in ("shapley_rmse", "banzhaf_rmse", "loo_rmse", "sii2_rmse", "test_rmse")}})
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame, by: list[str], cols: list[str], stat: str = "mean") -> pd.DataFrame:
    agg = getattr(df.groupby(by)[cols], stat)()
    return agg.reset_index()


def pivot(df: pd.DataFrame, index: str, columns: str, value: str) -> pd.DataFrame:
    return df.pivot_table(index=index, columns=columns, values=value, aggfunc="mean")
