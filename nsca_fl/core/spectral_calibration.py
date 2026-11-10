"""Step 2: MP bulk edge, Tracy-Widom correction, and TW-penetration margin.

Implements Eqs. (mp-singular-bound), (tw-fluctuation-scale), (4-mu-th),
(4-rho-th) and (tw-penetration) of the paper.
"""

from dataclasses import dataclass

import numpy as np
from scipy.stats import norm

# 95% quantile of the Tracy-Widom F_1 distribution (Johnstone, 2001).
TW_Q95 = 0.9793


@dataclass(frozen=True)
class SpectralThresholds:
    tau: float          # MP bulk edge for singular values
    lambda_tw: float    # relative eigenvalue-level TW tolerance
    mu_th: float        # eigenvalue threshold on K = G_tilde G_tilde^T
    rho_th: float       # singular-value detection threshold
    delta_tw: float     # TW-penetration margin delta_TW = rho_th - tau
    delta_min: float    # = delta_TW
    p_noise: float      # null activation probability 2*Phi(-kappa)
    p_thr: float        # rejection threshold at level alpha_c


def z_significance(alpha_c: float) -> float:
    """zeta_{1-alpha_c} = Phi^{-1}(1 - alpha_c)."""
    return float(norm.ppf(1.0 - alpha_c))


def mp_bulk_edge(sigma_eff: float, d: int, n: int) -> float:
    """tau = sigma_eff * (sqrt(d) + sqrt(N)), Eq. (mp-singular-bound)."""
    return sigma_eff * (np.sqrt(d) + np.sqrt(n))


def tw_fluctuation(gamma: float, n: int) -> float:
    """lambda_TW = q95 * sqrt(gamma)/(1+sqrt(gamma))^{2/3} * N^{-2/3}."""
    return TW_Q95 * (np.sqrt(gamma) / (1.0 + np.sqrt(gamma)) ** (2.0 / 3.0)) * n ** (-2.0 / 3.0)


def noise_activation(kappa: float) -> float:
    """p_noise = 2 * Phi(-kappa)  (null activation probability)."""
    return float(2.0 * norm.sf(kappa))


def rejection_threshold(kappa: float, n: int, alpha_c: float) -> float:
    """p_thr = p_noise + z * sqrt(p_noise (1-p_noise)/N), Eq. (p-thr)."""
    p_noise = noise_activation(kappa)
    return p_noise + z_significance(alpha_c) * np.sqrt(p_noise * (1.0 - p_noise) / n)


def compute_thresholds(cfg) -> SpectralThresholds:
    """Closed-form spectral calibration: no empirical estimation involved."""
    tau = mp_bulk_edge(cfg.sigma_eff, cfg.dim, cfg.n_clients)
    lambda_tw = tw_fluctuation(cfg.gamma, cfg.n_clients)
    mu_th = (cfg.sigma_eff ** 2) * (np.sqrt(cfg.dim) + np.sqrt(cfg.n_clients)) ** 2 * (1.0 + lambda_tw)
    rho_th = np.sqrt(mu_th)
    delta_tw = rho_th - tau
    return SpectralThresholds(
        tau=tau,
        lambda_tw=lambda_tw,
        mu_th=mu_th,
        rho_th=rho_th,
        delta_tw=delta_tw,
        delta_min=delta_tw,                    # Delta_min <- delta_TW  (Eq. tw-penetration)
        p_noise=noise_activation(cfg.kappa),
        p_thr=rejection_threshold(cfg.kappa, cfg.n_clients, cfg.alpha_c),
    )
