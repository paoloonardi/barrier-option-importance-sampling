"""Price the up-and-in put with all methods and save the sample-path figure.

Usage: python scripts/run_pricing.py [--paths 50000] [--seed 42]
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
    simulate_gbm_paths,
    up_and_in_put_continuous,
)
from params import PARAMS

FIG_DIR = Path(__file__).resolve().parents[1] / "figures"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--paths", type=int, default=50_000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    p = PARAMS

    cont = up_and_in_put_continuous(p["S0"], p["K"], p["r"], p["sigma"], p["T"], p["B"])
    bgk = bgk_corrected_price(**p)
    mc = price_standard_mc(**p, n_paths=args.paths, seed=args.seed)
    is_ = price_importance_sampling(**p, n_paths=args.paths, seed=args.seed + 1)

    print(f"Continuous-barrier price (Rubinstein-Reiner): {cont:.4f}")
    print(f"BGK-corrected price ({p['n_steps']} monitoring dates): {bgk:.4f}\n")
    print(f"| Metric          | Standard MC        | Importance sampling |")
    print(f"|-----------------|--------------------|---------------------|")
    print(f"| Price           | {mc.price:.4f}             | {is_.price:.4f}              |")
    print(f"| Standard error  | {mc.std_error:.5f}            | {is_.std_error:.5f}             |")
    print(f"| 95% CI          | [{mc.ci_low:.4f}, {mc.ci_high:.4f}] | [{is_.ci_low:.4f}, {is_.ci_high:.4f}]  |")
    print(f"| Knock-in rate   | {mc.knock_in_rate:.2%}             | {is_.knock_in_rate:.2%}              |")
    print(f"| Time (s)        | {mc.elapsed:.3f}              | {is_.elapsed:.3f}               |")
    vr = (mc.std_error / is_.std_error) ** 2
    print(f"\nSE ratio: {mc.std_error / is_.std_error:.2f}x | variance ratio: {vr:.1f}x"
          f" | time-adjusted efficiency gain: {is_.efficiency / mc.efficiency:.1f}x")

    # Sample paths figure
    S = simulate_gbm_paths(p["S0"], p["r"], p["sigma"], p["T"], p["n_steps"], 25, seed=7)
    t = np.linspace(0, p["T"], p["n_steps"] + 1)
    fig, ax = plt.subplots(figsize=(10, 6))
    for path in S:
        ax.plot(t, path, color="tab:red" if path.max() >= p["B"] else "tab:blue", lw=1)
    ax.axhline(p["B"], color="purple", ls="--", label="Barrier $B_u$")
    ax.axhline(p["K"], color="green", ls="--", label="Strike $K$")
    ax.set(xlabel="Time (years)", ylabel="Stock price",
           title="Simulated GBM paths (red = knocked in)")
    ax.legend(); ax.grid(alpha=0.3)
    FIG_DIR.mkdir(exist_ok=True)
    fig.savefig(FIG_DIR / "sample_paths.png", dpi=150, bbox_inches="tight")
    print(f"\nSaved {FIG_DIR / 'sample_paths.png'}")


if __name__ == "__main__":
    main()
