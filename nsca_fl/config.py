"""Hyperparameters of NSCA-FL (fixed constants of the paper, Sec. 5.1)."""

from dataclasses import dataclass, field


@dataclass
class NSCAConfig:
    # --- System model ---
    n_clients: int = 100                 # N
    dim: int = 20_000                    # d  (use >= N; keep modest for demo)
    sigma_eff: float = 1.0               # sigma_eff = sigma * C_clip

    # --- Fixed constants (paper Sec. 5.1: identical across settings) ---
    kappa: float = 1.5                   # activation threshold
    xi_th: float = 0.5                   # cross-round persistence threshold (A3)
    m_min: int = 5                       # minimum activation support
    alpha_c: float = 0.05                # significance level (TW tolerance)
    psi_est: float = 0.5                 # subspace-angle constant (A5)
    alpha_trim_cap: float = 0.49         # near-median fallback cap

    # --- Server ---
    eta: float = 1.0                     # server learning rate
    seed: int = 0

    # --- Derived -----------------------------------------------------------
    @property
    def gamma(self) -> float:
        """Aspect ratio gamma = N/d."""
        return self.n_clients / self.dim

    def alpha_trim(self, f: int) -> float:
        """Trimming fraction alpha_trim = min(f/N + 0.05, cap), Eq. setup."""
        return min(f / self.n_clients + 0.05, self.alpha_trim_cap)
