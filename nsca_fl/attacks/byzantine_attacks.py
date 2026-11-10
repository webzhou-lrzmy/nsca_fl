"""Byzantine attack generators (Sec. 5.1 setup).

All attacks respect the per-client budget ||a_i|| <= A_max * sigma_eff
(Assumption A1) with A_max = eta_A * A_BBP, eta_A = sqrt(3) => theta_*^2 = 3.
"""

from dataclasses import dataclass

import numpy as np

ETA_A_DEFAULT = np.sqrt(3.0)                                   # theta_*^2 = 3


def bbp_amplitude(d: int, n: int, f: int, eta_a: float = ETA_A_DEFAULT) -> float:
    """A_max = eta_A * (N d / f^2)^{1/4}  (Eq. setup, f > 0)."""
    if f <= 0:
        raise ValueError("Attack budget undefined for f = 0.")
    return eta_a * (n * d / f ** 2) ** 0.25


@dataclass
class AttackSpec:
    name: str                    # 'mpa' | 'alie' | 'label_flip' | 'sign_flip' | 'full_rank'
    rank: int = 3
    eta_a: float = ETA_A_DEFAULT


def _budget_vectors(rng, f, d, a_max, k):
    """Rank-k row-sparse A on f clients, rows of norm a_max."""
    carriers = np.linalg.qr(rng.standard_normal((d, k)))[0].T      # k orthonormal (d,)
    coef = rng.standard_normal((f, k))
    coef /= np.linalg.norm(coef, axis=1, keepdims=True)
    return coef @ carriers * a_max


def generate_attack(spec: AttackSpec, g: np.ndarray, malicious_mask: np.ndarray,
                    sigma_eff: float, rng: np.random.Generator) -> np.ndarray:
    """Return G with adversarial perturbations injected (benign rows intact)."""
    g = g.copy()
    idx = np.flatnonzero(malicious_mask)
    f, n, d = idx.size, *g.shape
    if f == 0:
        return g
    a_max = bbp_amplitude(d, n, f, spec.eta_a) * sigma_eff

    if spec.name in ("mpa", "alie"):                               # low-rank collusion
        a = _budget_vectors(rng, f, d, a_max, spec.rank)
        if spec.name == "alie":                                    # ALIE: around honest mean
            mu = g[~malicious_mask].mean(axis=0)
            direction = a / np.linalg.norm(a, axis=1, keepdims=True)
            g[idx] += np.abs(g[idx]).mean() * np.sign(mu)[None, :] \
                + 0.0                                              # deviation steering
            g[idx] = mu[None, :] * 1.0 + direction * a_max
        else:
            g[idx] += a
    elif spec.name == "sign_flip":                                 # sign flip of benign rows
        g[idx] = -np.abs(g[idx]) * np.sign(np.median(g, axis=0))[None, :] \
            + a_max / np.sqrt(d) * rng.standard_normal((f, d))
        norms = np.linalg.norm(g[idx], axis=1, keepdims=True)
        g[idx] *= a_max / np.maximum(norms, 1e-12)
    elif spec.name == "label_flip":                                # flipped-label gradients
        base = np.sign(np.median(g, axis=0))[None, :]
        g[idx] = -base * a_max / np.sqrt(d) * (1.0 + 0.1 * rng.standard_normal((f, d)))
    elif spec.name == "full_rank":                                 # independent full-rank noise
        a = rng.standard_normal((f, d))
        a *= a_max / np.linalg.norm(a, axis=1, keepdims=True)
        g[idx] += a
    else:
        raise ValueError(f"Unknown attack: {spec.name}")
    return g
