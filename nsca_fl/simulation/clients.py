"""Client update generation, Eq. (threat-decomposition):
g_i = s_i * b + e_i + a_i + n_i."""

import numpy as np


def benign_updates(n_benign: int, d: int, b: np.ndarray, sigma_eff: float,
                   e_scale: float, rng: np.random.Generator) -> np.ndarray:
    """Honest rows: b + e_i + n_i with ||e_i|| = O(sigma_eff) (Assumption A4)."""
    e = rng.standard_normal((n_benign, d)) * (e_scale / np.sqrt(d))
    noise = rng.standard_normal((n_benign, d)) * sigma_eff / np.sqrt(d)
    return b[None, :] + e + noise


def consensus_direction(d: int, rng: np.random.Generator) -> np.ndarray:
    v = rng.standard_normal(d)
    return v / np.linalg.norm(v)
