"""Attribution functionals of a fitted GP datamodel, with posterior uncertainty.

Every query used in the source (Shapley and Banzhaf values, leave-one-out and
joint-removal effects, Shapley and Banzhaf interaction indices, subset values)
is a linear functional L of the set function. Under a GP posterior, L[v] is
Gaussian: its mean is L applied to the posterior mean, its variance is
L L' k_post. Two routes are provided:

* dense: enumerate the 2^d coalitions (exact; used for d <= 14). The linear
  maps are obtained by applying the source's exact routines
  (tndm.attribution.exact) to the identity, so conventions match its evaluation.
* product: for any d, the posterior mean of Shapley, Banzhaf and leave-one-out
  values from the product form of the kernel sections,
  k(z, z_i) = sum_r C_ri prod_j u_rij(z_j). For a product function the
  Shapley value of j is (u_j(1) - u_j(0)) sum_s w(s) e_s, with e_s the
  coefficient of t^s in prod_{k != j} (u_k(0) + t u_k(1)) and
  w(s) = s!(d-1-s)!/d!; the Banzhaf value is (u_j(1) - u_j(0)) prod_{k != j}
  (u_k(0) + u_k(1)) / 2.
"""
from __future__ import annotations

from functools import lru_cache
from math import factorial

import numpy as np
import torch

from tndm.attribution import exact

from .coalitions import all_masks

__all__ = ["linear_map", "dense_posterior", "product_shapley", "product_banzhaf",
           "product_loo", "QUERIES"]

QUERIES = ("shapley", "banzhaf", "loo", "sii2", "bii2", "sii3")


@lru_cache(maxsize=32)
def linear_map(d: int, query: str) -> np.ndarray:
    """(n_items, 2^d) matrix A with A @ v = the query evaluated on the table v."""
    eye = np.eye(1 << d)
    if query == "shapley":
        out = exact.shapley_values(eye)
    elif query == "banzhaf":
        out = exact.banzhaf_values(eye)
    elif query == "loo":
        out = exact.loo_values(eye)
    elif query == "sii2":
        out = exact.pair_interactions(eye, shapley=True)
    elif query == "bii2":
        out = exact.pair_interactions(eye, shapley=False)
    elif query == "sii3":
        out = exact.triple_interactions(eye, shapley=True)
    else:
        raise ValueError(query)
    return np.ascontiguousarray(out)


def dense_posterior(model, query: str, output: int = 0, covariance: bool = True):
    """Exact posterior mean (and covariance) of a query by enumerating the cube."""
    d = model.d
    masks = all_masks(d)
    A = linear_map(d, query)
    mean = A @ model.predict(masks)[:, output]
    if not covariance:
        return mean, None
    cov = A @ model.posterior_cov(masks, output) @ A.T
    return mean, cov


# -- product route ----------------------------------------------------------------

def _shapley_size_weights(d: int) -> torch.Tensor:
    return torch.tensor([factorial(s) * factorial(d - 1 - s) / factorial(d) for s in range(d)])


def _poly_mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Multiply batched polynomials along the last axis (coefficients of t^0..)."""
    out = torch.zeros(*a.shape[:-1], a.shape[-1] + b.shape[-1] - 1)
    for k in range(b.shape[-1]):
        out[..., k:k + a.shape[-1]] += a * b[..., k:k + 1]
    return out


def _leave_one_out_polys(U0: torch.Tensor, U1: torch.Tensor) -> torch.Tensor:
    """For each j, coefficients of prod_{k != j} (U0_k + t U1_k).  Shapes (..., d) -> (..., d, d)."""
    d = U0.shape[-1]
    lead = U0.shape[:-1]
    prefix = [torch.ones(*lead, 1)]
    for k in range(d):
        prefix.append(_poly_mul(prefix[-1], torch.stack((U0[..., k], U1[..., k]), -1)))
    suffix = [torch.ones(*lead, 1)]
    for k in range(d - 1, -1, -1):
        suffix.append(_poly_mul(suffix[-1], torch.stack((U0[..., k], U1[..., k]), -1)))
    suffix = suffix[::-1]  # suffix[k] = prod_{m >= k}
    out = torch.zeros(*lead, d, d)
    for j in range(d):
        out[..., j, :] = _poly_mul(prefix[j], suffix[j + 1])
    return out


def _block(model, output: int):
    blk = next(b for b in model.blocks if output in b["cols"])
    return blk, blk["cols"].index(output)


@torch.no_grad()
def product_shapley(model, output: int = 0) -> np.ndarray:
    """Posterior mean of the Shapley values, in O(n R d^3), for any d."""
    blk, k = _block(model, output)
    C, U0, U1 = blk["model"].cov.factors(model.z)       # (R,n), (R,n,d), (R,n,d)
    polys = _leave_one_out_polys(U0, U1)                 # (R,n,d,d)
    w = _shapley_size_weights(model.d)
    weighted = (polys * w).sum(-1)                       # (R,n,d)
    per_point = (C[..., None] * (U1 - U0) * weighted).sum(0)   # (n,d)
    phi = per_point.T @ blk["alpha"][:, k]
    return phi.numpy() * model.sd[output]


@torch.no_grad()
def product_banzhaf(model, output: int = 0) -> np.ndarray:
    blk, k = _block(model, output)
    C, U0, U1 = blk["model"].cov.factors(model.z)
    half = 0.5 * (U0 + U1)
    log_half = torch.log(half)
    others = torch.exp(log_half.sum(-1, keepdim=True) - log_half)   # prod_{k != j}
    per_point = (C[..., None] * (U1 - U0) * others).sum(0)
    return (per_point.T @ blk["alpha"][:, k]).numpy() * model.sd[output]


@torch.no_grad()
def product_loo(model, output: int = 0) -> np.ndarray:
    """v(N) - v(N minus j) under the posterior mean."""
    blk, k = _block(model, output)
    C, U0, U1 = blk["model"].cov.factors(model.z)
    log_u1 = torch.log(U1)
    others = torch.exp(log_u1.sum(-1, keepdim=True) - log_u1)
    per_point = (C[..., None] * (U1 - U0) * others).sum(0)
    return (per_point.T @ blk["alpha"][:, k]).numpy() * model.sd[output]
