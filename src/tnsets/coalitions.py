"""Coalition encodings.

A coalition S of d training groups is an integer bitmask: bit i is set when
group i is in S. ``bits_of(masks, d)`` gives the (n, d) inclusion matrix z with
z[n, i] = 1 iff group i is in coalition n. These are the conventions of the
source's release (third_party/TNDM/src/tndm/utils/coalitions.py).
"""
from __future__ import annotations

import numpy as np


def bits_of(masks, d: int) -> np.ndarray:
    """Inclusion matrix of integer masks; an (n, d) 0/1 matrix is passed through (needed for d > 62)."""
    array = np.asarray(masks)
    if array.ndim == 2:
        if array.shape[1] != d:
            raise ValueError(f"inclusion matrix has {array.shape[1]} columns, expected {d}")
        return array.astype(np.float64)
    masks = array.astype(np.int64)
    return ((masks[:, None] >> np.arange(d)) & 1).astype(np.float64)


def masks_of(bits) -> np.ndarray:
    bits = np.asarray(bits, dtype=np.int64)
    return bits @ (np.int64(1) << np.arange(bits.shape[1], dtype=np.int64))


def all_masks(d: int) -> np.ndarray:
    return np.arange(1 << d, dtype=np.int64)


def popcount(masks) -> np.ndarray:
    masks = np.asarray(masks, dtype=np.int64)
    out = np.zeros(masks.shape, dtype=np.int64)
    work = masks.copy()
    while np.any(work):
        out += work & 1
        work >>= 1
    return out
