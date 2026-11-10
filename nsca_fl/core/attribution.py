"""Step 3: noise-calibrated subspace attribution (Eq. attribution-stats).

Gates: activation support m_j, activation ratio p_j, sign coherence nu_j,
TW-penetration margin Delta_j, and cross-round persistence xi_j.
Routing rules (a)/(b)/(c) of the paper.
"""

from dataclasses import dataclass
from enum import Enum

import numpy as np

from .spectral_calibration import z_significance


class Routing(Enum):
    SUB_BULK = "sub_bulk"      # rho_j <= tau           (beta = 1)
    SAFE = "safe"              # rule (a)               (beta = 1; structurally unreachable)
    EXCISE = "excise"          # rule (b), Tier-1       (beta = 0)
    CLAMP = "clamp"            # rule (c) / borderline  (beta = tau / rho_j)


@dataclass
class AttributionResult:
    routing: Routing
    support_idx: np.ndarray    # J_j
    m_j: int
    p_j: float
    nu_j: float
    nu_thr: float
    delta_j: float
    xi_j: float


def activation_stats(u_j: np.ndarray, kappa: float) -> tuple:
    """J_j = {i : |u_{j,i}| > kappa / sqrt(N)}, p_j = |J_j|/N, and nu_j."""
    n = u_j.shape[0]
    support = np.abs(u_j) > kappa / np.sqrt(n)                 # sign(y_i)=sign(u_{j,i}), rho_j>0
    m_j = int(support.sum())
    p_j = m_j / n
    if m_j == 0:
        return support, m_j, p_j, 0.0
    y = u_j[support]
    median_sign = np.sign(np.median(y))
    coherent = np.sign(y) == median_sign
    nu_j = float(coherent.sum()) / m_j
    return support, m_j, p_j, nu_j


def sign_coherence_threshold(m_j: int, alpha_c: float) -> float:
    """nu_thr = 1/2 + zeta_{1-alpha_c} / (2 sqrt(m_j)), Eq. (nu-thr)."""
    if m_j <= 0:
        return np.inf
    return 0.5 + z_significance(alpha_c) / (2.0 * np.sqrt(m_j))


def persistence_statistic(v_j: np.ndarray, prev_excised: list) -> float:
    """xi_j = max_{j'} |v_j^T v_{j'}^prev|; 0 when history is empty (t = 1)."""
    prev = [p for p in prev_excised if p is not None]
    if not prev:
        return 0.0
    return float(max(abs(float(v_j @ p)) for p in prev))
def classify_candidate(
    m_j: int, p_j: float, nu_j: float, delta_j: float, xi_j: float,
    rho_j: float, tau: float, rho_th: float, delta_min: float,
    p_thr: float, cfg,
) -> Routing:
    """Routing of one candidate direction per Step 3 rules (a)-(c)."""
    if rho_j <= tau:
        return Routing.SUB_BULK
    if rho_j < rho_th:
        return Routing.CLAMP                                   # TW borderline band
    # --- Candidate (rho_j >= rho_th) ---
    if m_j < cfg.m_min or delta_j < delta_min:                 # Algorithm 1 line 11
        return Routing.CLAMP
    nu_thr = sign_coherence_threshold(m_j, cfg.alpha_c)
    # Rule (a): majority-coherent protrusion -> safe retention (unreachable in theory).
    if m_j >= cfg.m_min and nu_j >= nu_thr and p_j >= 0.5 and xi_j < cfg.xi_th:
        return Routing.SAFE
    # Rule (b): low-rank contamination -> Tier-1 excision.
    if m_j >= cfg.m_min and p_thr < p_j < 0.5 and nu_j >= nu_thr and delta_j >= delta_min:
        return Routing.EXCISE
    # Rule (c): everything else -> Tier-2 clamping.
    return Routing.CLAMP

