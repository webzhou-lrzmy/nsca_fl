"""Step 1: median-anchored sign alignment and implicit truncated SVD.

Implements Eq. (sign-alignment) and Eq. (implicit-svd).
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class AlignedSpectrum:
    g_tilde: np.ndarray        # sign-aligned matrix (N x d)
    d_sign: np.ndarray         # diagonal of D_sign (length N)
    eigvals: np.ndarray        # mu_j, descending
    sing_vals: np.ndarray      # rho_j = sqrt(mu_j)
    left_vecs: np.ndarray      # u_j (N x r)


def median_anchored_alignment(g: np.ndarray) -> tuple:
    """Canonical row alignment: g_tilde = D_sign G with sign(g_i^T m)."""
    m = np.median(g, axis=0)                                   # coordinate-wise median
    dots = g @ m
    flip = np.sum(dots < 0) > g.shape[0] / 2                   # invert m if majority negative
    if flip:
        m, dots = -m, -dots
    d_sign = np.where(dots < 0, -1.0, 1.0)                     # sign(0) := +1
    g_tilde = d_sign[:, None] * g
    return g_tilde, d_sign, m


def implicit_truncated_svd(g_tilde: np.ndarray, tau: float) -> AlignedSpectrum:
    """Gram-based eigendecomposition; stop at first rho_j < tau (r_cut)."""
    k_gram = g_tilde @ g_tilde.T                               # N x N (client Gram)
    mu, u = np.linalg.eigh(k_gram)                             # ascending -> reverse
    order = np.argsort(mu)[::-1]
    mu, u = mu[order], u[:, order]
    mu = np.clip(mu, 0.0, None)
    rho = np.sqrt(mu)
    r_cut = int(np.sum(rho >= tau))                            # #{j : rho_j >= tau}
    return AlignedSpectrum(
        g_tilde=g_tilde,
        d_sign=None,
        eigvals=mu[:r_cut],
        sing_vals=rho[:r_cut],
        left_vecs=u[:, :r_cut],
    )
