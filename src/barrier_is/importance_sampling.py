"""Two-phase drift-tilting importance sampling for the up-and-in put.

Log-increments under the pricing measure Q:
    dX_t ~ N(mu, sigma^2 dt),  mu = (r - sigma^2/2) dt.
Under the sampling measure the mean becomes mu + sigma^2 dt * theta, with
theta = theta_plus before the first barrier crossing tau (inclusive) and
theta = theta_minus afterwards.

Per-step likelihood ratio dQ/dQ~ = exp(-theta * dX + psi(theta)),
psi(theta) = mu*theta + sigma^2 dt theta^2 / 2.

Because theta_plus/minus = a +/- d with a = 1/2 - r/sigma^2, and
psi(theta) = sigma^2 dt/2 * ((theta - a)^2 - a^2), we get psi(theta_plus) =
psi(theta_minus) =: psi. The path likelihood ratio therefore collapses to
    log L = (theta_minus - theta_plus) X_tau - theta_minus X_T + n psi.
"""

import time

import numpy as np

from .monte_carlo import MCResult, _batches, summarise


def optimal_tilts(S0, K, r, sigma, T, B):
    """Return (theta_plus, theta_minus)."""
    b = np.log(B / S0)
    c = np.log(S0 / K)
    a = 0.5 - r / sigma**2
    d = (2 * b + c) / (T * sigma**2)
    return a + d, a - d


def price_importance_sampling(
    S0, K, r, sigma, T, B, n_steps, n_paths, seed=None, batch_size=20_000
) -> MCResult:
    """IS estimator, vectorised (no per-path Python loop)."""
    t0 = time.perf_counter()
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    mu = (r - 0.5 * sigma**2) * dt
    vol = sigma * np.sqrt(dt)
    b = np.log(B / S0)
    disc = np.exp(-r * T)

    th_p, th_m = optimal_tilts(S0, K, r, sigma, T, B)
    psi = mu * th_m + 0.5 * sigma**2 * dt * th_m**2
    shift_p = mu + sigma**2 * dt * th_p
    shift_m = mu + sigma**2 * dt * th_m

    payoffs = np.empty(n_paths)
    hit = np.empty(n_paths, dtype=bool)
    for start, k in _batches(n_paths, batch_size):
        noise = vol * rng.standard_normal((k, n_steps))
        rows = np.arange(k)

        # Phase 1: whole path under theta_plus, used up to the first crossing.
        X_plus = np.cumsum(shift_p + noise, axis=1)
        crossed = X_plus >= b
        h = crossed.any(axis=1)
        tau = crossed.argmax(axis=1)  # first crossing index (meaningless if not h)
        X_tau = X_plus[rows, tau]

        # Phase 2: same shocks, theta_minus drift on steps tau+1, ..., n-1.
        C_minus = np.cumsum(shift_m + noise, axis=1)
        X_T = X_tau + C_minus[:, -1] - C_minus[rows, tau]

        log_lr = np.where(h, (th_m - th_p) * X_tau - th_m * X_T + n_steps * psi, 0.0)
        put = np.maximum(K - S0 * np.exp(X_T), 0.0)
        payoffs[start : start + k] = np.where(h, disc * put * np.exp(log_lr), 0.0)
        hit[start : start + k] = h
    return summarise(payoffs, hit, t0)
