#!/usr/bin/env python3
"""Build the paper's tables and in-text numbers from the run outputs and released source results.

    python scripts/make_tables.py            # writes paper/tables/*.tex and results/tables/*.csv

Every number in the paper's results comes from here (paper/tables/*.tex, and
paper/tables/numbers.tex for numbers quoted in the text); none is typed by hand.
The runs are read from results/runs/, or from $TNSETS_RUNS (src/tnsets/report.py).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from tnsets import report  # noqa: E402

OUT = ROOT / "paper" / "tables"
CSV = ROOT / "results" / "tables"
OUT.mkdir(parents=True, exist_ok=True)
CSV.mkdir(parents=True, exist_ok=True)
NUMBERS: dict[str, str] = {}


DIGITS = ("Zero", "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine")


def number(name: str, value: str) -> None:
    """Record \\<name> for tables/numbers.tex; digits are spelled out (LaTeX names are letters only)."""
    NUMBERS["".join(DIGITS[int(c)] if c.isdigit() else c for c in name)] = value


def fmt(x: float, digits: int = 3) -> str:
    if x is None or not np.isfinite(x):
        return "--"
    if x == 0:
        return "0"
    if abs(x) >= 100:
        return f"{x:.0f}"
    mag = int(np.floor(np.log10(abs(x))))
    decimals = max(0, digits - 1 - mag)
    return f"{x:.{decimals}f}"


def cell(mean: float, sd: float | None = None, bold: bool = False, digits: int = 3) -> str:
    text = fmt(mean, digits)
    if bold:
        text = f"\\textbf{{{text}}}"
    if sd is not None and np.isfinite(sd):
        text += f"\\,{{\\scriptsize$\\pm${fmt(sd, 1)}}}"
    return text


def write(name: str, body: str) -> None:
    (OUT / f"{name}.tex").write_text(body)
    print(f"wrote tables/{name}.tex")


def best_index(values: list[float], higher: bool = False) -> set[int]:
    finite = [(v, i) for i, v in enumerate(values) if v is not None and np.isfinite(v)]
    if not finite:
        return set()
    target = max(finite)[0] if higher else min(finite)[0]
    return {i for v, i in finite if np.isclose(v, target, rtol=1e-9, atol=0)}


# ------------------------------------------------------------------ d = 24
D24_ROWS = [("source:linear", "Linear"), ("source:degree2_ridge", "Degree-2 ridge"),
            ("source:degree3_ridge", "Degree-3 ridge"), ("source:degree3_omp", "Degree-3 OMP"),
            ("source:rbf_krr", "RBF kernel ridge"), ("source:ht", "HT"),
            ("source:ladder_tt", "TT degree ladder"), ("source:cp", "TNDM-CP"),
            ("source:tt", "TNDM-TT"), ("gp", "GP datamodel (ours)")]


def table_d24() -> pd.DataFrame:
    t = report.d24_table()
    t.to_csv(CSV / "d24.csv", index=False)
    cols = [("nmse", 256), ("nmse", 512), ("nmse", 1024), ("lro_rmse", 1024), ("loo_rmse", 1024)]
    stats = {}
    for key, _ in D24_ROWS:
        sub = t[t.method == key]
        stats[key] = [(sub[sub.budget == b][c].mean(), sub[sub.budget == b][c].std()) for c, b in cols]
    lines = []
    for j in range(len(cols)):
        winners = best_index([stats[k][j][0] for k, _ in D24_ROWS])
        for i, (k, _) in enumerate(D24_ROWS):
            stats[k][j] = (*stats[k][j], i in winners)
    for k, label in D24_ROWS:
        cells = [cell(m, s, b) for m, s, b in stats[k]]
        lines.append(f"{label} & " + " & ".join(cells) + " \\\\")
        if k == "source:tt":
            lines.append("\\midrule")
    body = ("\\begin{tabular}{@{}lccccc@{}}\n\\toprule\n"
            "& \\multicolumn{3}{c}{Test NMSE} & \\multicolumn{2}{c}{Removal RMSE, $n=1024$} \\\\\n"
            "\\cmidrule(lr){2-4}\\cmidrule(lr){5-6}\n"
            "Datamodel & $n=256$ & $n=512$ & $n=1024$ & joint & LOO \\\\\n\\midrule\n"
            + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n")
    write("d24", body)
    def mean(method: str, col: str, budget: int) -> float:
        sub = t[(t.method == method) & (t.budget == budget)][col]
        return float(sub.mean()) if len(sub) else float("nan")

    for b in (256, 512, 1024):
        gp_b, tt_b, cp_b = (mean(k, "nmse", b) for k in ("gp", "source:tt", "source:cp"))
        number(f"dTwentyFourGPnmse{b}", fmt(gp_b))
        number(f"dTwentyFourTTnmse{b}", fmt(tt_b))
        number(f"dTwentyFourCPnmse{b}", fmt(cp_b))
        number(f"dTwentyFourRatio{b}", f"{min(tt_b, cp_b) / gp_b:.0f}" if np.isfinite(gp_b) else "--")
        number(f"dTwentyFourGPlro{b}", fmt(mean("gp", "lro_rmse", b)))
        number(f"dTwentyFourTTlro{b}", fmt(mean("source:tt", "lro_rmse", b)))
        number(f"dTwentyFourGPsupport{b}", f"{100 * mean('gp', 'support_recovery', b):.0f}")
    number("dTwentyFourGPloo", fmt(mean("gp", "loo_rmse", 1024)))
    number("dTwentyFourTTloo", fmt(mean("source:tt", "loo_rmse", 1024)))
    return t


# ------------------------------------------------------------------ curves (figure data)
def curves() -> None:
    """Per-budget summaries for the figure: d = 8 NMSE, d = 12 AP, modular and AG News errors."""
    d8 = report.d8_table()
    d8.to_csv(CSV / "d8.csv", index=False)
    d8.groupby(["method", "budget"])[["nmse", "shapley_rel_l2", "sii_rel_l2"]].median().reset_index() \
        .to_csv(CSV / "curve_d8.csv", index=False)
    d12 = report.d12_table()
    d12.to_csv(CSV / "d12.csv", index=False)
    d12.groupby(["method", "budget"])[["complete_interaction_ap", "top1_complete", "top8_complete_fraction",
                                       "shapley_selection_complete", "banzhaf_selection_complete"]] \
        .mean().reset_index().to_csv(CSV / "curve_d12.csv", index=False)
    attr = report.attribution_table()
    attr.to_csv(CSV / "attribution.csv.gz", index=False)
    mod = attr[attr.cube.str.startswith("modular") & (attr.output_name == "triple_ABC")]
    mod = mod.groupby(["method", "budget", "cube"])[report.ATTR_COLS].mean().groupby(["method", "budget"]).mean()
    mod.reset_index().to_csv(CSV / "curve_modular.csv", index=False)
    ag = attr[attr.cube.str.startswith("agnews")]
    ag.groupby(["method", "budget"])[report.ATTR_COLS].median().reset_index().to_csv(CSV / "curve_agnews.csv", index=False)

    c8 = d8.groupby(["method", "budget"]).nmse.median()
    c12 = d12.groupby(["method", "budget"]).complete_interaction_ap.mean()
    for b in (48, 64, 128):
        number(f"dEightGP{b}", fmt(c8["gp"][b]))
    number("dEightBestBaselineAtOneTwoEight", fmt(min(c8[k][128] for k in c8.index.levels[0] if k.startswith("source:"))))
    for b in (32, 64, 96, 160):
        number(f"dTwelveGPap{b}", fmt(c12["gp"][b]))
        number(f"dTwelveTTap{b}", fmt(c12["source:tt"][b]))
        number(f"dTwelveBestap{b}", fmt(max(c12[k][b] for k in c12.index.levels[0]
                                            if k.startswith("source:") and np.isfinite(c12[k].get(b, np.nan)))))
    agm = ag.groupby(["method", "budget"]).median(numeric_only=True)
    for col, tag in (("shapley_rel_l2", "Shapley"), ("banzhaf_rel_l2", "Banzhaf")):
        number(f"agNewsGP{tag}OneTwoEight", fmt(agm.loc[("gp", 128), col]))
        number(f"agNewsTT{tag}OneTwoEight", fmt(agm.loc[("source:tt", 128), col]))
    modm = mod
    number("modularGPShapleySixtyFour", fmt(modm.loc[("gp", 64), "shapley_rel_l2"]))
    number("modularCPShapleySixtyFour", fmt(modm.loc[("source:cp", 64), "shapley_rel_l2"]))
    number("modularGPsiiSixtyFour", fmt(modm.loc[("gp", 64), "sii_rel_l2"]))
    number("modularCPsiiSixtyFour", fmt(modm.loc[("source:cp", 64), "sii_rel_l2"]))


# ------------------------------------------------------------------ Sport-Music and broader
def table_sportmusic_broader() -> None:
    sm = report.sportmusic_table()
    sm.to_csv(CSV / "sportmusic.csv", index=False)
    paper = {r.method: (r.r2, r.sd) for r in sm.itertuples()}
    gp_units = report.load_units("sportmusic-gp-all")
    gp = (gp_units.r2.mean(), gp_units.r2.std(ddof=0))
    rows = [("Linear (tuned)", paper["paper:linear (tuned)"]), ("Matched-pair (tuned)", paper["paper:matched (tuned)"]),
            ("Degree-2 (tuned)", paper["paper:degree-2 (tuned)"]), ("TNDM-CP, rank 5", paper["paper:CP single fit, R=5"]),
            ("TNDM-CP, ensemble of 8", paper["paper:CP ENSEMBLE ranks 3-6"]), ("GP datamodel (ours)", gp)]
    winners = best_index([r[1][0] for r in rows], higher=True)
    body = "\\begin{tabular}{@{}lc@{}}\n\\toprule\nDatamodel & Held-out $R^2$ \\\\\n\\midrule\n"
    for i, (label, (m, s)) in enumerate(rows):
        body += f"{label} & {cell(m, s, i in winners)} \\\\\n"
    body += "\\bottomrule\n\\end{tabular}\n"
    write("sportmusic", body)
    number("smGP", fmt(gp[0])); number("smCP", fmt(paper["paper:CP single fit, R=5"][0]))
    number("smCPens", fmt(paper["paper:CP ENSEMBLE ranks 3-6"][0])); number("smLinear", fmt(paper["paper:linear (tuned)"][0]))

    b = report.broader_table()
    b.to_csv(CSV / "broader.csv", index=False)
    methods = [("source:als_rbf", "RBF kernel ridge"), ("source:als_degree3_ridge", "Degree-3 ridge"),
               ("source:als_cp", "TNDM-CP (ALS)"), ("source:als_tt", "TNDM-TT (ALS)"),
               ("source:grad_cp", "TNDM-CP (gradient)"), ("source:grad_tt", "TNDM-TT (gradient)"),
               ("gp", "GP datamodel (ours)")]
    cubes = [("atlas_ag_news_topics_d10", "AG News topics"), ("atlas_uci_har_clients_d10", "UCI HAR"),
             ("atlas_qwen_coin_icl_d8", "Coin flip")]
    for metric, name in (("nmse", "broader"), ("nmse_no_empty", "broader_noempty")):
        med = b.groupby(["cube", "method", "m"])[metric].median()
        cols = [(c, m) for c, _ in cubes for m in sorted(b[b.cube == c].m.unique())]
        vals = {k: [med.get((c, k, m), np.nan) for c, m in cols] for k, _ in methods}
        win = [best_index([vals[k][j] for k, _ in methods]) for j in range(len(cols))]
        head = " & ".join(f"\\multicolumn{{3}}{{c}}{{{label}}}" for _, label in cubes)
        rule = "".join(f"\\cmidrule(lr){{{2 + 3 * i}-{4 + 3 * i}}}" for i in range(len(cubes)))
        ms = " & ".join(f"{m}" for _, m in cols)
        body = ("\\begin{tabular}{@{}l" + "c" * len(cols) + "@{}}\n\\toprule\n& " + head + " \\\\\n" + rule + "\n"
                "$n$ & " + ms + " \\\\\n\\midrule\n")
        for i, (k, label) in enumerate(methods):
            body += label + " & " + " & ".join(cell(vals[k][j], None, i in win[j]) for j in range(len(cols))) + " \\\\\n"
        body += "\\bottomrule\n\\end{tabular}\n"
        write(name, body)
    # the coin-flip cube, where BIC keeps one component: two components fitted directly
    coin = b[(b.cube == "atlas_qwen_coin_icl_d8") & (b.method == "gp-r2direct")].groupby("m").nmse.median()
    number("coinTwoComp64", fmt(coin[64])); number("coinTwoComp128", fmt(coin[128]))


# ------------------------------------------------------------------ attribution against exact games
def table_attribution() -> None:
    p = report.probe_table()
    p.to_csv(CSV / "probe.csv", index=False)
    cols = ["shapley_rmse", "banzhaf_rmse", "loo_rmse", "sii2_rmse"]
    rows = {"qwen_probe": [("source:linear", "Linear"), ("source:rbf_kernel", "RBF kernel ridge"),
                           ("source:orderwise_cp", "TNDM-CP"), ("source:tensor_train", "TNDM-TT"), ("gp", "GP datamodel (ours)")],
            "uci_har": [("source:als_linear", "Linear"), ("source:als_rbf", "RBF kernel ridge"),
                        ("source:als_cp", "TNDM-CP"), ("source:als_tt", "TNDM-TT"), ("gp", "GP datamodel (ours)")]}
    means = {}
    for shard, entries in rows.items():
        sub = p[p.shard == shard]
        for k, _ in entries:
            means[(shard, k)] = [1e3 * sub[sub.method == k][c].mean() for c in cols]
    body = ("\\begin{tabular}{@{}lcccccccc@{}}\n\\toprule\n& \\multicolumn{4}{c}{Qwen probe (3 seeds)} & "
            "\\multicolumn{4}{c}{UCI HAR (10 splits)} \\\\\n\\cmidrule(lr){2-5}\\cmidrule(lr){6-9}\n"
            "Datamodel & Shapley & Banzhaf & LOO & SII & Shapley & Banzhaf & LOO & SII \\\\\n\\midrule\n")
    labels = [label for _, label in rows["qwen_probe"]]
    win = {}
    for shard, entries in rows.items():
        for j in range(4):
            win[(shard, j)] = best_index([means[(shard, k)][j] for k, _ in entries])
    for i, label in enumerate(labels):
        kq, kh = rows["qwen_probe"][i][0], rows["uci_har"][i][0]
        cells = [cell(means[("qwen_probe", kq)][j], None, i in win[("qwen_probe", j)]) for j in range(4)]
        cells += [cell(means[("uci_har", kh)][j], None, i in win[("uci_har", j)]) for j in range(4)]
        body += label + " & " + " & ".join(cells) + " \\\\\n"
    body += "\\bottomrule\n\\end{tabular}\n"
    write("attribution", body)
    q = {k: means[("qwen_probe", k)] for k, _ in rows["qwen_probe"]}
    number("probeGPShapley", fmt(q["gp"][0])); number("probeRBFShapley", fmt(q["source:rbf_kernel"][0]))
    number("probeTTShapley", fmt(q["source:tensor_train"][0]))
    number("probeRatioRBF", f"{q['source:rbf_kernel'][0] / q['gp'][0]:.0f}")
    number("probeRatioTT", f"{q['source:tensor_train'][0] / q['gp'][0]:.0f}")


# ------------------------------------------------------------------ ablations
VARIANTS = [("gp", "GP datamodel (BIC order)"), ("gp-r1", "One component"),
            ("gp-stationary", "One stationary component"), ("gp-iso", "Isotropic (one relevance)"),
            ("gp-flat", "Flat priors"), ("gp-r2direct", "Two components, fresh starts only"),
            ("gp-tau0.5", "$\\tau=0.5$"), ("gp-tau2", "$\\tau=2$")]


def table_ablation() -> None:
    d24 = report.d24_table()
    d8 = report.d8_table()
    d12 = report.d12_table()
    attr = pd.read_csv(CSV / "attribution.csv.gz")
    mod = attr[attr.cube.str.startswith("modular") & (attr.output_name == "triple_ABC")]
    ag = attr[attr.cube.str.startswith("agnews")]
    br = pd.read_csv(CSV / "broader.csv")
    probe = pd.read_csv(CSV / "probe.csv")
    sm = {m: report.load_units(f"sportmusic-{m}-all") for m, _ in VARIANTS}
    columns = [
        ("R24, NMSE, $n=256$", lambda k: d24[(d24.method == k) & (d24.budget == 256)].nmse.mean(), False),
        ("R24, NMSE, $n=1024$", lambda k: d24[(d24.method == k) & (d24.budget == 1024)].nmse.mean(), False),
        ("R8, NMSE, $n=48$", lambda k: d8[(d8.method == k) & (d8.budget == 48)].nmse.median(), False),
        ("R12, AP, $n=96$", lambda k: d12[(d12.method == k) & (d12.budget == 96)].complete_interaction_ap.mean(), True),
        ("Modular, Shapley, $n=32$", lambda k: mod[(mod.method == k) & (mod.budget == 32)]
            .groupby("cube").shapley_rel_l2.mean().mean(), False),
        ("AG News, Shapley, $n=128$", lambda k: ag[(ag.method == k) & (ag.budget == 128)].shapley_rel_l2.median(), False),
        ("Sport--Music, $R^2$", lambda k: sm[k].r2.mean() if len(sm[k]) else np.nan, True),
        ("UCI HAR, NMSE, $n=512$", lambda k: br[(br.cube == "atlas_uci_har_clients_d10") & (br.method == k)
            & (br.m == 512)].nmse.median(), False),
        ("Probe, Shapley ($\\times10^{3}$)", lambda k: 1e3 * probe[(probe.shard == "qwen_probe") & (probe.method == k)]
            .shapley_rmse.mean(), False),
    ]
    vals = {k: [fn(k) for _, fn, _ in columns] for k, _ in VARIANTS}
    win = [best_index([vals[k][j] for k, _ in VARIANTS], higher=columns[j][2]) for j in range(len(columns))]
    head = " & ".join(f"\\rotatebox{{90}}{{{label}}}" for label, _, _ in columns)
    body = "\\begin{tabular}{@{}l" + "c" * len(columns) + "@{}}\n\\toprule\nVariant & " + head + " \\\\\n\\midrule\n"
    for i, (k, label) in enumerate(VARIANTS):
        body += label + " & " + " & ".join(cell(vals[k][j], None, i in win[j]) for j in range(len(columns))) + " \\\\\n"
    body += "\\bottomrule\n\\end{tabular}\n"
    write("ablation", body)
    pd.DataFrame(vals, index=[c[0] for c in columns]).to_csv(CSV / "ablation.csv")
    # how far the BIC-selected order is from the best variant of each column
    ratios = []
    for j, (_, _, higher) in enumerate(columns):
        col = [vals[k][j] for k, _ in VARIANTS]
        best = max(col) if higher else min(col)
        ratios.append(best / vals["gp"][j] if higher else vals["gp"][j] / best)
    ratios = np.asarray(ratios)
    number("ablationColumns", str(len(columns)))
    number("ablationBICnear", str(int((ratios <= 1.15).sum())))
    number("ablationBICworst", f"{np.ceil(10 * ratios.max()) / 10:.1f}")    # "at most": round up
    worst_fixed = {k: max((vals[k][j] / min(vals[kk][j] for kk, _ in VARIANTS)) if not columns[j][2]
                          else (max(vals[kk][j] for kk, _ in VARIANTS) / vals[k][j]) for j in range(len(columns)))
                   for k, _ in VARIANTS}
    fixed_min = min(v for k, v in worst_fixed.items() if k != 'gp')
    number("ablationFixedWorstMin", f"{np.floor(10 * fixed_min) / 10:.1f}")    # "at least": round down


# ------------------------------------------------------------------ calibration
def table_calibration() -> None:
    rows = []
    for run in sorted(report.RUNS.glob("calibration-*")):
        config = json.loads((run / "config.json").read_text())
        for unit, m in report._units_of(run).items():
            for q, res in m["outputs"].items():
                for query, v in res.items():
                    if isinstance(v, dict) and not query.endswith("_cv"):
                        rows.append({"method": config["method"], "shard": config["shard"], "budget": m["budget"],
                                     "query": query, **{k: v[k] for k in ("cover50", "cover80", "cover90", "cover95", "mean_z2")}})
    c = pd.DataFrame(rows)
    c.to_csv(CSV / "calibration.csv", index=False)
    family = {"modular": "Modular", "agnews": "AG News LoRA", "atlas_ag": "AG News topics", "atlas_uci": "UCI HAR",
              "atlas_qwen": "Coin flip", "qwen_probe": "Qwen probe", "khop": "Reasoning 8"}

    def fam(shard: str) -> str:
        return next(v for k, v in family.items() if shard.startswith(k))

    empty = c[c.shard.str.endswith(":empty")]
    c = c[~c.shard.str.endswith(":empty")].copy()
    c["family"] = c.shard.map(fam)
    sub = c[c.method == "gp"]
    if len(empty):   # observing the empty coalition (it needs no training) does not repair the intervals
        e = empty[(empty.method == "gp") & (empty["query"] == "shapley")]
        number("calEmptyShapleyNinety", fmt(e.cover90.mean(), 2))
        base = sub[sub.shard.isin(empty.shard.str.removesuffix(":empty").unique()) & (sub["query"] == "shapley")]
        number("calNoEmptyShapleyNinety", fmt(base.cover90.mean(), 2))
    order = list(dict.fromkeys(family.values()))
    queries = [("heldout", "Held-out outcomes"), ("shapley", "Shapley values"), ("sii2", "Pairwise SII")]
    body = ("\\begin{tabular}{@{}l" + "ccc" * len(queries) + "@{}}\n\\toprule\n& "
            + " & ".join(f"\\multicolumn{{3}}{{c}}{{{label}}}" for _, label in queries) + " \\\\\n"
            + "".join(f"\\cmidrule(lr){{{2 + 3 * i}-{4 + 3 * i}}}" for i in range(len(queries))) + "\n"
            "Cubes & " + " & ".join(["50\\%", "90\\%", "$\\bar z^2$"] * len(queries)) + " \\\\\n\\midrule\n")
    for f in order:
        cells = []
        for q, _ in queries:
            s = sub[(sub.family == f) & (sub["query"] == q)]
            cells += [fmt(s.cover50.mean(), 2), fmt(s.cover90.mean(), 2), fmt(s.mean_z2.median(), 2)]
        body += f"{f} & " + " & ".join(cells) + " \\\\\n"
    body += "\\bottomrule\n\\end{tabular}\n"
    write("calibration", body)
    shap = sub[sub["query"] == "shapley"]
    number("calShapleyNinety", fmt(shap.cover90.mean(), 2))
    held = sub[sub["query"] == "heldout"]
    number("calHeldoutNinety", fmt(held.cover90.mean(), 2))


# ------------------------------------------------------------------ cost
def table_cost() -> None:
    d24 = report.d24_table()
    src = report.load_units("d24-source-")
    rows = []
    for _, r in src.iterrows():
        for fam, m in r["families"].items():
            rows.append({"family": fam, "budget": r["budget"], "seed": r["seed"], "fit_seconds": m["fit_seconds"],
                         "nmse": m["nmse"]})
    s = pd.DataFrame(rows)
    s.to_csv(CSV / "d24_source_refit.csv", index=False)
    gp = d24[d24.method == "gp"].groupby("budget").fit_seconds.mean()
    for b in (256, 512, 1024):
        number(f"costGP{b}", f"{gp[b]:.0f}" if b in gp else "--")
        for fam in ("tt", "cp"):
            sub = s[(s.family == fam) & (s.budget == b)]
            number(f"cost{fam.upper()}{b}", f"{sub.fit_seconds.mean():.0f}")
    # reproduction check of the refits against the frozen predictions
    frozen = d24[d24.method.isin(["source:tt", "source:cp"])].groupby(["method", "budget"]).nmse.mean()
    refit = s[s.family.isin(["tt", "cp"])].groupby(["family", "budget"]).nmse.mean()
    gap = max(abs(refit[(f, b)] - frozen[(f"source:{f}", b)]) for f in ("tt", "cp") for b in (256, 512, 1024))
    number("refitGapExponent", str(int(np.ceil(np.log10(max(gap, 1e-16))))))


# ------------------------------------------------------------------ further in-text numbers
def text_numbers() -> None:
    d12 = pd.read_csv(CSV / "curve_d12.csv").set_index(["method", "budget"])
    src12 = [k for k in d12.index.levels[0] if k.startswith("source:")]
    for b in (128, 160):
        number(f"dTwelveGPtopEight{b}", fmt(d12.loc[("gp", b), "top8_complete_fraction"], 2))
        number(f"dTwelveBesttopEight{b}", fmt(max(d12.loc[(k, b), "top8_complete_fraction"] for k in src12
                                                if (k, b) in d12.index and np.isfinite(d12.loc[(k, b), "top8_complete_fraction"])), 2))
        number(f"dTwelveGPshapleySel{b}", f"{100 * d12.loc[('gp', b), 'shapley_selection_complete']:.0f}")
        number(f"dTwelveBestshapleySel{b}", f"{100 * max(d12.loc[(k, b), 'shapley_selection_complete'] for k in src12 if (k, b) in d12.index and np.isfinite(d12.loc[(k, b), 'shapley_selection_complete'])):.0f}")
    ag = pd.read_csv(CSV / "curve_agnews.csv").set_index(["method", "budget"])
    for b in (16, 32, 64, 128):
        number(f"agNewsGPnmse{b}", fmt(ag.loc[("gp", b), "heldout_nmse"]))
        number(f"agNewsLinearnmse{b}", fmt(ag.loc[("source:linear", b), "heldout_nmse"]))
    # selected model orders of the proposed model, per benchmark
    orders = []
    for prefix, label in (("d24-gp-seed", "Reasoning 24"), ("d12-gp-seed", "Reasoning 12"), ("d8-gp-all", "Reasoning 8"),
                          ("attribution-gp-modular", "Modular"), ("attribution-gp-agnews", "AG News LoRA"),
                          ("sportmusic-gp-all", "Sport--Music"), ("broader-gp-atlas", "Broader"), ("probe-gp-", "Attribution")):
        u = report.load_units(prefix)
        if not len(u):
            continue
        u = u[u.method == "gp"]
        if "order" in u:
            per_unit = list(u["order"])
        else:   # d = 24 units record the two parts separately
            per_unit = [list(zip(c, a)) for c, a in zip(u["components"], u["amplitude"])]
        for o in per_unit:
            for rank, amp in o:
                orders.append({"benchmark": label, "rank": rank, "amplitude": amp})
    o = pd.DataFrame(orders)
    o.to_csv(CSV / "orders.csv", index=False)
    share = lambda mask: f"{100 * mask.mean():.0f}"
    number("orderStationaryShare", share((o["rank"] == 1) & ~o.amplitude))
    number("orderMultiShare", share(o["rank"] >= 2))
    for label, key in (("Reasoning 24", "RTwentyFour"), ("Reasoning 8", "REight"), ("Broader", "Broader"),
                       ("AG News LoRA", "AGNews"), ("Attribution", "Probe")):
        sub = o[o.benchmark == label]
        if len(sub):
            number(f"order{key}Stationary", share((sub["rank"] == 1) & ~sub.amplitude))
            number(f"order{key}Multi", share(sub["rank"] >= 2))


def scaling_numbers() -> None:
    """Attribution without enumeration at d = 100 (scripts/scaling_demo.py)."""
    sc = json.loads((CSV / "scaling.json").read_text())
    number("scaleD", str(sc["d"]))
    number("scaleN", f"{sc['n']:,}".replace(",", "{,}"))
    number("scaleFitSeconds", f"{sc['fit_seconds']:.0f}")
    number("scaleShapleySeconds", f"{sc['seconds']['shapley']:.2f}")
    number("scaleOtherMax", fmt(sc["shapley_other_max_abs"], 2))
    if not sc["shapley_top6_is_support"]:
        raise RuntimeError("scaling demo: the six largest Shapley values are not the support")


def paired_tests() -> None:
    """Paired comparisons of the GP datamodel with TNDM-TT and TNDM-CP on identical designs."""
    from scipy.stats import wilcoxon
    tns = ("source:tt", "source:cp")
    d8 = pd.read_csv(CSV / "d8.csv")
    shares, pvals = [], []
    for b in (32, 48, 64, 128):
        s = d8[d8.budget == b].pivot_table(index="repeat", columns="method", values="nmse")
        for tn in tns:
            shares.append((s["gp"] < s[tn]).mean())
            pvals.append(wilcoxon(s["gp"], s[tn]).pvalue)
    number("pairDEightWinMin", f"{100 * min(shares):.0f}")
    number("pairDEightWinPhrase", "every one of the 20 samples" if min(shares) == 1.0
           else f"at least {100 * min(shares):.0f}\\% of the 20 samples")
    number("pairDEightPExp", str(int(np.ceil(np.log10(max(pvals))))))
    attr = pd.read_csv(CSV / "attribution.csv.gz")
    ag = attr[attr.cube.str.startswith("agnews")]
    shares, pvals = [], []
    for b in (16, 32, 64, 128):
        s = ag[ag.budget == b].pivot_table(index=["cube", "repeat"], columns="method", values="shapley_rel_l2")
        for tn in tns:
            ok = s[["gp", tn]].dropna()
            shares.append((ok["gp"] < ok[tn]).mean())
            pvals.append(wilcoxon(ok["gp"], ok[tn]).pvalue)
    number("pairAGNewsWinMin", f"{100 * min(shares):.0f}")
    number("pairAGNewsWinMax", f"{100 * max(shares):.0f}")
    number("pairAGNewsPExp", str(int(np.ceil(np.log10(max(pvals))))))
    mod = attr[attr.cube.str.startswith("modular") & (attr.output_name == "triple_ABC")]
    s = mod[mod.budget == 32].pivot_table(index=["cube", "repeat"], columns="method", values="shapley_rel_l2")
    number("pairModularCPThirtyTwoP", fmt(wilcoxon(s["gp"], s["source:cp"]).pvalue, 2))


def more_numbers() -> None:
    """Numbers quoted in the text that are not in a table."""
    ag = pd.read_csv(CSV / "curve_agnews.csv")
    number("agNewsBestNMSEOneTwoEight", fmt(ag[ag.budget == 128].heldout_nmse.min(), 2))
    cal = pd.read_csv(CSV / "calibration.csv")
    cal = cal[(cal.method == "gp") & ~cal.shard.str.endswith(":empty")]
    good = cal[cal.shard.str.startswith(("modular", "agnews", "khop")) & (cal["query"] == "shapley")]
    by = good.assign(fam=good.shard.str.split("_").str[0]).groupby("fam").cover90.mean()
    number("calGoodShapleyLow", fmt(by.min(), 2))
    number("calGoodShapleyHigh", fmt(by.max(), 2))
    bad = cal[cal.shard.str.startswith("atlas") & (cal["query"] == "shapley")]
    number("calBadShapleyHigh", fmt(bad.groupby("shard").cover90.mean().max(), 2))
    probe = pd.read_csv(CSV / "probe.csv")
    q = probe[probe.shard == "qwen_probe"].groupby("method").shapley_rmse.mean()
    number("probeStationaryRatio", f"{q['gp-stationary'] / q['gp']:.0f}")
    blob = np.load(ROOT / "third_party/TNDM/data/cubes/andor_d24_k2_g3.npz")
    meta = json.loads(str(blob["meta_json"]))
    seconds = float(meta["cost_per_coalition"].split()[1])      # "median 47.6 GPU-seconds (...)"
    number("retrainSecondsPerCoalition", f"{seconds:.1f}")
    number("retrainHoursOneZeroTwoFour", f"{seconds * 1024 / 3600:.1f}")
    d24 = pd.read_csv(CSV / "d24.csv")
    for k, tag in (("source:tt", "TT"), ("source:cp", "CP"), ("gp", "GP")):
        number(f"dTwentyFour{tag}completeTwoFiveSix",
               fmt(d24[(d24.method == k) & (d24.budget == 256)].top3_size3_completeness.mean(), 2))
    br = pd.read_csv(CSV / "broader.csv")
    med = br.groupby(["cube", "m", "method"]).nmse_no_empty.median().unstack("method")
    keep = [c for c in med.columns if c.startswith("source:") or c == "gp"]
    wins = int((med[keep].idxmin(axis=1) == "gp").sum())
    number("broaderNoEmptyWins", str(wins))
    number("broaderNoEmptySettings", str(len(med)))


def main() -> None:
    table_d24()
    curves()
    text_numbers()
    table_sportmusic_broader()
    table_attribution()
    table_ablation()
    table_calibration()
    table_cost()
    more_numbers()
    paired_tests()
    scaling_numbers()
    (OUT / "numbers.tex").write_text("".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in sorted(NUMBERS.items())))
    print(f"wrote tables/numbers.tex ({len(NUMBERS)} numbers)")


if __name__ == "__main__":
    main()
