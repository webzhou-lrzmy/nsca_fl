"""NSCA-FL demo: run Algorithm 1 on synthetic Byzantine-attacked updates.

Usage:
    python main.py            # full demo (d = 20000)
    python main.py --quick    # fast demo (d = 6000)
"""

import argparse

from nsca_fl.config import NSCAConfig
from nsca_fl.simulation.experiment import run_persistence, run_single_round

BASELINES = ["NSCA-FL", "FedAvg", "Median", "TrimmedMean", "Krum"]


def report_single(cfg):
    print("=" * 70)
    print(f"Single-round MPA (k=3) attack | N={cfg.n_clients}, d={cfg.dim}, "
          f"sigma_eff={cfg.sigma_eff}")
    print("=" * 70)
    res = run_single_round(cfg, attack_name="mpa")
    print(f"{'Method':<14}{'cos-sim to benign consensus b':>34}")
    print("-" * 48)
    for name in BASELINES:
        print(f"{name:<14}{res[name]:>34.4f}")
    print("-" * 48)
    print(f"r_cut = {res['r_cut']} | excised update mass = "
          f"{res['excised_mass']:.3f} | flagged malicious = "
          f"{res['flagged_malicious']}/{res['f']}")


def report_persistence(cfg):
    print("\n" + "=" * 70)
    print("Multi-round persistence: xi_j statistics (persistent MPA subspace)")
    print("=" * 70)
    for h in run_persistence(cfg, n_rounds=4, rotating=False):
        print(f"round {h['round']}: max xi_j = {h['xi_max']:.3f} | "
              f"r_cut = {h['r_cut']} | excised mass = {h['excised_mass']:.3f}")


def main():
    parser = argparse.ArgumentParser(description="NSCA-FL demo")
    parser.add_argument("--quick", action="store_true", help="smaller dimension")
    args = parser.parse_args()

    cfg = NSCAConfig(dim=6_000 if args.quick else 20_000)
    report_single(cfg)
    report_persistence(cfg)


if __name__ == "__main__":
    main()
