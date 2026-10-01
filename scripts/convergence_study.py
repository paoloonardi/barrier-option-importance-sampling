"""Convergence of both estimators with 95% confidence bands.

Usage: python scripts/convergence_study.py [--max-paths 50000] [--step 1000]
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from barrier_is import (
    bgk_corrected_price,
    price_importance_sampling,
    price_standard_mc,
    up_and_in_put_continuous,
)
from barrier_is.monte_carlo import Z_95
from params import PARAMS

FIG_DIR = Path(__file__).resolve().parents[1] / "figures"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-paths", type=int, default=50_000)
    ap.add_argument("--step", type=int, default=1_000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    p = PARAMS

    sizes = np.arange(args.step, args.max_paths + 1, args.step)
    est = {"Standard MC": [], "Importance sampling": []}
    for i, m in enumerate(sizes):
        est["Standard MC"].append(price_standard_mc(**p, n_paths=m, seed=args.seed + 2 * i))
        est["Importance sampling"].append(
            price_importance_sampling(**p, n_paths=m, seed=args.seed + 2 * i + 1))

    fig, ax = plt.subplots(figsize=(9, 5.5))
    for label, res in est.items():
        mean = np.array([x.price for x in res])
        se = np.array([x.std_error for x in res])
        ax.plot(sizes, mean, label=label)
        ax.fill_between(sizes, mean - Z_95 * se, mean + Z_95 * se, alpha=0.2)
    cont = up_and_in_put_continuous(p["S0"], p["K"], p["r"], p["sigma"], p["T"], p["B"])
    ax.axhline(cont, color="red", ls="--", label="Continuous-barrier price")
    ax.axhline(bgk_corrected_price(**p), color="green", ls="--", label="BGK-corrected price")
    ax.set(xlabel="Number of paths", ylabel="Estimated option price",
           title="Convergence of Monte Carlo estimators (95% CI)")
    ax.legend(); ax.grid(alpha=0.3)
    FIG_DIR.mkdir(exist_ok=True)
    fig.savefig(FIG_DIR / "convergence.png", dpi=150, bbox_inches="tight")
    print(f"Saved {FIG_DIR / 'convergence.png'}")


if __name__ == "__main__":
    main()
