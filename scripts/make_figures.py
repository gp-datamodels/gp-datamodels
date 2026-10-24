#!/usr/bin/env python3
"""Figures of the paper, from results/tables/*.csv (written by scripts/make_tables.py).

    python scripts/make_figures.py      # writes paper/figures/*.pdf

Colours: the first five slots of the dataviz reference palette, in its
validated order, each series keeping its slot in every panel; markers and line
styles carry the identity too, for grey-scale print and colour-vision deficiency.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CSV = ROOT / "results" / "tables"
FIG = ROOT / "paper" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
SERIES = {  # entity -> (label, colour, marker, linestyle)
    "gp": ("GP datamodel (ours)", "#2a78d6", "o", "-"),
    "source:tt": ("TNDM-TT", "#eb6834", "s", "--"),
    "source:cp": ("TNDM-CP", "#1baf7a", "^", "--"),
    "rbf": ("RBF kernel ridge", "#eda100", "D", ":"),
    "deg3": ("Degree-3 ridge", "#e87ba4", "v", ":"),
}
ALIASES = {"rbf": ("source:rbf", "source:rbf_kernel", "source:rbf_kernel_ridge"),
           "deg3": ("source:degree3_ridge",)}

plt.rcParams.update({"font.size": 8, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.linewidth": 0.6, "font.family": "serif", "pdf.fonttype": 42})


def series(df: pd.DataFrame, entity: str, col: str):
    names = ALIASES.get(entity, (entity,))
    sub = df[df.method.isin(names)].sort_values("budget")
    sub = sub[np.isfinite(sub[col])]
    return sub.budget.to_numpy(), sub[col].to_numpy()


def panel(ax, df, col, entities, title, ylabel, log=True, ylim=None, logx=True, yticks=None):
    for e in entities:
        x, y = series(df, e, col)
        if len(x) == 0:
            continue
        label, colour, marker, ls = SERIES[e]
        yy = y.copy()
        clipped = np.zeros(len(y), bool)
        if ylim is not None:   # values outside the axis are drawn on its edge, hollow
            clipped = (yy > ylim[1]) | (yy < ylim[0])
            yy = np.clip(yy, ylim[0], ylim[1])
        lw = 1.6 if e == "gp" else 1.1
        ax.plot(x, yy, color=colour, lw=lw, ls=ls, zorder=3 if e == "gp" else 2)
        ax.scatter(x[~clipped], yy[~clipped], color=colour, marker=marker, s=22, zorder=4,
                   edgecolors="white", linewidths=0.6, label=label)
        if clipped.any():
            ax.scatter(x[clipped], yy[clipped], facecolors="white", edgecolors=colour, marker=marker, s=22,
                       zorder=4, linewidths=1.0)
    if logx:
        ax.set_xscale("log", base=2)
    if log:
        ax.set_yscale("log")
        ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
        ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    if ylim is not None:
        ax.set_ylim(*ylim)
    if yticks is not None:
        ax.set_yticks(yticks)
        ax.set_yticklabels([f"{v:g}" for v in yticks])
    budgets = sorted(df.budget.unique())
    ax.set_xticks(budgets)
    ax.set_xticklabels([str(int(b)) for b in budgets])
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.grid(True, color=GRID, lw=0.5, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.set_title(title, fontsize=8.5, color=INK, loc="left")
    ax.set_xlabel("observed coalitions $n$")
    ax.set_ylabel(ylabel)


def figure_curves() -> None:
    d8 = pd.read_csv(CSV / "curve_d8.csv")
    d12 = pd.read_csv(CSV / "curve_d12.csv")
    mod = pd.read_csv(CSV / "curve_modular.csv")
    ag = pd.read_csv(CSV / "curve_agnews.csv")
    fig, grid = plt.subplots(2, 2, figsize=(6.6, 4.6), constrained_layout=True)
    axes = grid.ravel()
    every = ["gp", "source:tt", "source:cp", "rbf", "deg3"]
    panel(axes[0], d8, "nmse", every, "(a) Reasoning 8: held-out NMSE", "NMSE", ylim=(0.03, 4),
          yticks=[0.05, 0.1, 0.2, 0.5, 1, 2, 4])
    panel(axes[1], d12, "complete_interaction_ap", ["gp", "source:tt", "rbf", "deg3"],
          "(b) Reasoning 12: AP of complete triples", "average precision", log=False, ylim=(0, 1.02), logx=False)
    panel(axes[2], mod, "shapley_rel_l2", every, "(c) Modular: Shapley error", "relative $\\ell_2$ error",
          ylim=(1e-3, 4))
    panel(axes[3], ag, "shapley_rel_l2", every, "(d) AG News LoRA: Shapley error", "relative $\\ell_2$ error",
          ylim=(0.2, 4), yticks=[0.2, 0.3, 0.5, 1, 2, 4])
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=3, frameon=False, fontsize=7.5)
    fig.savefig(FIG / "curves.pdf")
    plt.close(fig)
    print("wrote figures/curves.pdf")


def figure_relevance() -> None:
    """Relevance profile of each output at d = 24, seed 31, m = 256 (refitted here: deterministic)."""
    sys.path.insert(0, str(ROOT / "src"))
    from tnsets import methods
    from tnsets.benchmarks import d24
    values = d24.cube(31)
    train = np.asarray(d24.read("gpu_training_masks.txt")[:256], dtype=np.int64)
    y = np.vstack([values[int(m)] for m in train])
    model = methods.fit("gp", train, y, d24.D)
    rel = np.vstack([model.group_relevance(q) for q in range(d24.Q)])
    rel = rel / rel.max(1, keepdims=True)
    fig, ax = plt.subplots(figsize=(7.2, 2.0), constrained_layout=True)
    image = ax.imshow(rel, cmap=matplotlib.colors.LinearSegmentedColormap.from_list(
        "seq", ["#f7fafe", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]), aspect="auto", vmin=0, vmax=1)
    for q, roles in enumerate(d24.PROVIDERS):
        for role in roles:
            for g in role:
                ax.add_patch(plt.Rectangle((g - 0.5, q - 0.5), 1, 1, fill=False, edgecolor=INK, lw=0.9))
    ax.set_xticks(range(d24.D))
    ax.set_yticks(range(d24.Q))
    ax.set_yticklabels([f"{q + 1}" for q in range(d24.Q)])
    ax.set_xlabel("training group")
    ax.set_ylabel("relation sequence")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    bar = fig.colorbar(image, ax=ax, fraction=0.02, pad=0.01)
    bar.set_label("relevance (row max = 1)", fontsize=7)
    bar.outline.set_linewidth(0.4)
    fig.savefig(FIG / "relevance.pdf")
    plt.close(fig)
    np.savetxt(CSV / "relevance_d24_seed31_m256.csv", rel, delimiter=",", fmt="%.6f")
    print("wrote figures/relevance.pdf")


if __name__ == "__main__":
    figure_curves()
    figure_relevance()
