"""Standard (crude) Monte Carlo for the discretely monitored up-and-in put."""

import time
from dataclasses import dataclass

import numpy as np

Z_95 = 1.959963984540054


@dataclass(frozen=True)
class MCResult:
    price: float
    std_error: float
    ci_low: float
    ci_high: float
    knock_in_rate: float  # fraction of simulated paths that hit the barrier
    n_paths: int
    elapsed: float  # wall-clock seconds

    @property
    def efficiency(self) -> float:
        """Inverse of (variance per path x time per path); higher is better."""
        var = self.std_error**2 * self.n_paths
        return 1.0 / (var * self.elapsed / self.n_paths)


def summarise(discounted_payoffs, hit, t0) -> MCResult:
    m = discounted_payoffs.size
    price = float(discounted_payoffs.mean())
    se = float(discounted_payoffs.std(ddof=1) / np.sqrt(m))
    return MCResult(
        price=price,
        std_error=se,
        ci_low=price - Z_95 * se,
        ci_high=price + Z_95 * se,
        knock_in_rate=float(hit.mean()),
        n_paths=m,
        elapsed=time.perf_counter() - t0,
    )


def _batches(n_paths, batch_size):
    for start in range(0, n_paths, batch_size):
        yield start, min(batch_size, n_paths - start)


def simulate_gbm_paths(S0, r, sigma, T, n_steps, n_paths, seed=None):
    """GBM paths on the grid t_1, ..., t_n (S0 prepended as column 0)."""
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    inc = (r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * rng.standard_normal(
        (n_paths, n_steps)
    )
    logS = np.log(S0) + np.cumsum(inc, axis=1)
    return np.hstack([np.full((n_paths, 1), S0), np.exp(logS)])


def price_standard_mc(
    S0, K, r, sigma, T, B, n_steps, n_paths, seed=None, batch_size=20_000
) -> MCResult:
    """Crude MC estimator, fully vectorised and processed in batches."""
    t0 = time.perf_counter()
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    drift = (r - 0.5 * sigma**2) * dt
    vol = sigma * np.sqrt(dt)
    b = np.log(B / S0)
    disc = np.exp(-r * T)

    payoffs = np.empty(n_paths)
    hit = np.empty(n_paths, dtype=bool)
    for start, k in _batches(n_paths, batch_size):
        X = np.cumsum(drift + vol * rng.standard_normal((k, n_steps)), axis=1)
        h = X.max(axis=1) >= b
        S_T = S0 * np.exp(X[:, -1])
        payoffs[start : start + k] = np.where(h, disc * np.maximum(K - S_T, 0.0), 0.0)
        hit[start : start + k] = h
    return summarise(payoffs, hit, t0)
