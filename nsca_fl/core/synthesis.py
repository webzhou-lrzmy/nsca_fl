"""Step 4: three-tier synthesis (Eq. matrix-clean-synthesis) and
coordinate-wise trimmed-mean aggregation (Lemma subthreshold)."""

import numpy as np


def beta_weights(routing_per_dir: list, rho: np.ndarray, tau: float) -> np.ndarray:
    """Per-direction retention coefficients beta_j, Eq. (cleansing-operator)."""
    from .attribution import Routing
    beta = np.ones_like(rho, dtype=float)
    for idx, route in enumerate(routing_per_dir):
        if route is Routing.EXCISE:
            beta[idx] = 0.0                                    # Tier-1
        elif route in (Routing.CLAMP, Routing.SUB_BULK):
            beta[idx] = min(1.0, tau / rho[idx])               # Tier-2 (sub-bulk: rho<=tau -> 1)
        # Routing.SAFE -> beta = 1
    return beta


def cleanse(g_tilde: np.ndarray, left_vecs: np.ndarray, sing_vals: np.ndarray,
            beta: np.ndarray) -> np.ndarray:
    """G_clean = G_tilde * sum_j beta_j v_j v_j^T (only over processed dirs)."""
    corr = left_vecs * (beta * sing_vals)                      # scaled columns
    return g_tilde - corr @ left_vecs.T                        # G_tilde (I - sum (1-beta) v v^T)


def trimmed_mean(x: np.ndarray, alpha_trim: float) -> np.ndarray:
    """Coordinate-wise trimmed mean: discard floor(alpha*N) extremes per side."""
    n = x.shape[0]
    k = int(np.floor(alpha_trim * n))
    if 2 * k >= n:
        raise ValueError("alpha_trim too large: no entries retained.")
    xs = np.sort(x, axis=0)
    return xs[k:n - k].mean(axis=0)
