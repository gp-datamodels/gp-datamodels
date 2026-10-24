# Gaussian-process datamodels for training data attribution

Code, run outputs and result tables for the paper *Gaussian-process datamodels
with presence-aware product covariances for training data attribution*.

A datamodel learns, from a few retraining runs, the set function that maps a
subset of training groups (a coalition) to the outcome of training on it, and
answers attribution queries from it: subset predictions, Shapley and Banzhaf
values, interaction indices, removal effects. The proposed datamodel is a
Gaussian process whose covariance is a sum of products, over training groups,
of 2×2 blocks indexed by the presence or absence of each group, each with a
relevance and a presence-dependent amplitude. One such product is the limit of a
CP tensor network datamodel (TNDM) with random factors as its rank grows. The
hyperparameters are fitted on all observed coalitions by maximizing the marginal
likelihood under a hierarchical prior, the number of products is chosen by BIC,
and every attribution query has a closed-form Gaussian posterior. Shapley,
Banzhaf and leave-one-out values are computed by product formulas, without
enumerating the 2^d coalitions.

The evaluation reuses the benchmark suite of TNDM (Heidari et al.,
[arXiv:2610.24721](https://arxiv.org/abs/2610.24721)): its released retraining
outcomes, designs, splits and saved results. A few results, from the tables in
`paper/tables/`:

| benchmark | GP datamodel | best TNDM or baseline |
|---|---|---|
| reasoning, 24 groups: test NMSE from 256 / 1,024 coalitions | 0.0178 / 0.00657 | TNDM-TT 0.472 / 0.0712 |
| reasoning, 12 groups: AP of complete triples, 160 coalitions | 0.984 | TNDM-TT 0.762 |
| Sport–Music (Qwen2-7B fine-tuning): held-out R² | 0.791 | TNDM-CP ensemble 0.734 |
| Qwen probe: Shapley RMSE (×10³) against exact values | 0.759 | RBF kernel ridge 9.97 |

The paper also reports where the approach falls short: on the coin-flip cube
TNDM-CP predicts better at 64 and 128 coalitions, and the posterior intervals are
narrower than their nominal level.

## Contents

| path | what |
|---|---|
| `src/tnsets/gp.py` | the GP datamodel: covariance, hyperpriors, MAP-II fit, BIC model order, posterior, closed-form leave-one-out |
| `src/tnsets/attribution.py` | attribution posteriors: exact by enumeration (small d) and by product formulas (any d) |
| `src/tnsets/methods.py` | the model and its variants as named presets (`gp`, `gp-r1`, `gp-stationary`, `gp-iso`, `gp-flat`, `gp-r2direct`, `gp-tau0.5`, `gp-tau2`) |
| `src/tnsets/benchmarks/` | the source's protocols on its released cubes, one module per benchmark |
| `src/tnsets/report.py` | collects run outputs and the source's saved results into tables |
| `scripts/` | `run_bench.py` (one run), `run_task.py` (one line of a campaign), `make_campaign.py`, `make_tables.py`, `make_figures.py`, `scaling_demo.py`, `get_tndm.sh` |
| `campaigns/` | the task lists of every run in the paper (359 runs) |
| `results/runs/` | the outputs of those runs: `config.json`, `env.json` (library versions only) and `metrics.json` (one entry per unit of work) |
| `results/tables/` | the tidy result tables built from them (CSV) |
| `paper/` | the paper's generated tables (LaTeX) and figures (PDF) |
| `tests/` | unit tests of the covariance, the fit and the attribution formulas |

## Installation

```bash
git clone https://github.com/gp-datamodels/gp-datamodels && cd gp-datamodels
bash scripts/get_tndm.sh               # the TNDM release at the commit the paper used
python -m venv .venv && . .venv/bin/activate
pip install -e third_party/TNDM        # cubes loaders, exact attribution routines, source fitters
pip install -e ".[dev]"
pytest
```

The runs of the paper used Python 3.12, NumPy 2.2 and PyTorch 2.6, in double
precision on CPUs; no GPU is needed, so the CPU build of PyTorch is enough.

## Usage

```python
import numpy as np
from tnsets import methods
from tnsets.attribution import dense_posterior, product_shapley

d, n = 10, 400
rng = np.random.default_rng(0)
z = rng.integers(0, 2, size=(n, d))               # observed coalitions: the groups each retrained model saw
y = z[:, :3].max(1) * z[:, 3:6].max(1)            # outcome: an AND of two ORs over groups 0-5
y = y + 0.02 * rng.standard_normal(n)             # retraining noise

model = methods.fit("gp", z, y, d)                # hyperparameters by MAP-II, model order by BIC
phi = product_shapley(model)                      # Shapley values, without enumerating the 2^d coalitions
mean, cov = dense_posterior(model, "shapley")     # exact posterior mean and covariance (small d only)
print(np.round(phi, 3))                           # close to 1/6 for groups 0-5 and to 0 for groups 6-9
print(np.round(np.sqrt(np.diag(cov)), 3))         # posterior standard deviations
```

Coalitions are rows of a 0/1 inclusion matrix, or integer bitmasks (bit j set
when group j is in the coalition) for d ≤ 62. `model.predict(masks)` and
`model.predict_var(masks)` give the posterior of the set function on any
coalitions, `product_banzhaf` and `product_loo` the other values, and
`dense_posterior(model, query)` any query of `attribution.linear_map`
(`shapley`, `banzhaf`, `loo`, `sii2`, `bii2`, `sii3`) with its covariance.

## Reproducing the paper

**Tables and figures from the shipped runs.** With the TNDM release in
`third_party/TNDM` (it holds the source's saved results that the baselines'
columns come from):

```bash
python scripts/make_tables.py      # results/tables/*.csv and paper/tables/*.tex
python scripts/make_figures.py     # paper/figures/*.pdf
```

| paper | file |
|---|---|
| Table 2, 24-group reasoning task | `paper/tables/d24.tex` |
| Table 3, Sport–Music | `paper/tables/sportmusic.tex` |
| Table 4, broader settings | `paper/tables/broader.tex` |
| Table 5, attribution accuracy | `paper/tables/attribution.tex` |
| Table 6, ablations | `paper/tables/ablation.tex` |
| Table 7, calibration | `paper/tables/calibration.tex` |
| Table B.8, broader settings without the empty coalition | `paper/tables/broader_noempty.tex` |
| numbers quoted in the text | `paper/tables/numbers.tex` |
| Figure 2, relevance profiles | `paper/figures/relevance.pdf` |
| Figure 3, accuracy against the number of coalitions | `paper/figures/curves.pdf` |
| attribution at d = 100 (Section 6.7) | `results/tables/scaling.json`, from `python scripts/scaling_demo.py` |

**Rerunning the experiments.** Each line of a campaign list is one run, a
(benchmark, method, shard) triple whose units of work (budgets, repetitions,
seeds) are checkpointed in its `metrics.json`, so an interrupted run resumes
where it stopped:

```bash
python scripts/run_task.py campaigns/final_rest.tsv 0                                # one run, into out/runs/
seq 0 323 | xargs -P 8 -I{} python scripts/run_task.py campaigns/final_rest.tsv {}   # a whole list
```

The four lists are `final_d24.tsv` and `final_rest.tsv` (the proposed model, its
variants and the calibration study), `source.tsv` (the source's released fitters,
rerun where it saved no result for the exact protocol) and
`calibration_empty.tsv`; `scripts/make_campaign.py` writes them. On a Slurm
cluster, `run_task.py` without a line index runs line `$SLURM_ARRAY_TASK_ID`
(see its docstring). The 359 runs took about 130 CPU core-hours, three
quarters of them for the 24-group task; each fit used four threads (eight for
the 24-group task), set by `--threads` or `$SLURM_CPUS_PER_TASK`. Once every list
has run, build the tables from the new runs instead of the shipped ones:

```bash
TNSETS_RUNS=out/runs python scripts/make_tables.py
```

Timings, and hence the cost columns, depend on the machine.

## Data and licenses

The retraining outcomes (27 coalition cubes), the source's saved predictions and
per-repetition results, and its fitters come from the TNDM release,
<https://github.com/TNDM-data-attribution/TNDM>, which `scripts/get_tndm.sh`
clones at commit `b20e0eb`. It is not redistributed here and is not covered by
this repository's license: at that commit it carries no license of its own, so
its use is subject to its authors' terms. The code and results in this
repository are released under the MIT License (`LICENSE`).
