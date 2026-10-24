"""Named method presets: the proposed datamodel and its ablations.

``fit(name, masks, y, d)`` returns an object with ``predict(masks) -> (n, Q)``.
"""
from __future__ import annotations

import time
from dataclasses import replace

from .gp import GPConfig, GPDatamodel

PRESETS = {
    # the proposed model: up to four presence-aware product components, the
    # number chosen by BIC, hierarchical priors (GPConfig defaults)
    "gp": GPConfig(rank=4),
    # ablations
    "gp-r1": GPConfig(rank=1),                                      # one component
    "gp-stationary": GPConfig(rank=1, amplitude=False),             # weighted-Hamming (ARD) GP
    "gp-iso": GPConfig(rank=1, ard=False, amplitude=False),         # one relevance: RBF-type GP
    "gp-flat": GPConfig(rank=4, tau=100.0, amp_sd=100.0),           # no shrinkage of the relevances
    "gp-r2direct": GPConfig(rank=2, extra={"strategy": "direct"}),  # two components, fresh starts only
    "gp-tau0.5": GPConfig(rank=4, tau=0.5),
    "gp-tau2": GPConfig(rank=4, tau=2.0),
}


def config(name: str, **overrides) -> GPConfig:
    base = PRESETS[name]
    return replace(base, **overrides) if overrides else base


def fit(name: str, masks, y, d: int, **overrides):
    start = time.perf_counter()
    model = GPDatamodel(d, config(name, **overrides)).fit(masks, y)
    model.fit_seconds = time.perf_counter() - start
    return model


def order(model) -> list[list]:
    """Selected model order per output block: [number of components, amplitude on]."""
    return [[blk["model"].cov.rank, blk["model"].cov.b is not None] for blk in model.blocks]
