"""Baseline server-side aggregators for comparison."""

import numpy as np

from ..core.synthesis import trimmed_mean


def fedavg(g: np.ndarray) -> np.ndarray:
    return g.mean(axis=0)


def coordinate_median(g: np.ndarray) -> np.ndarray:
    return np.median(g, axis=0)


def dp_trimmed_mean(g: np.ndarray, f: int) -> np.ndarray:
    return trimmed_mean(g, min(f / g.shape[0] + 0.05, 0.49))


def krum(g: np.ndarray, f: int) -> np.ndarray:
    """Classical Multi-Krum (m = 1 selection)."""
    n = g.shape[0]
    dist = np.linalg.norm(g[:, None, :] - g[None, :, :], axis=-1)
    np.fill_diagonal(dist, np.inf)
    scores = np.sort(dist, axis=1)[:, : n - f - 1].sum(axis=1)
    return g[int(np.argmin(scores))]
