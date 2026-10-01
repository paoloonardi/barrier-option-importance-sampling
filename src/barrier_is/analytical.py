"""Closed-form benchmarks under Black-Scholes.

* Continuously monitored up-and-in put (Rubinstein & Reiner, 1991).
* Broadie-Glasserman-Kou (1997) continuity correction for discrete monitoring.
"""

import numpy as np
from scipy.special import zeta
from scipy.stats import norm

#: beta = -zeta(1/2) / sqrt(2*pi) ~= 0.5826 (Broadie, Glasserman & Kou, 1997)
BGK_BETA = float(-zeta(0.5) / np.sqrt(2.0 * np.pi))


def vanilla_put(S0, K, r, sigma, T):
    """Black-Scholes price of a European put."""
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S0 * norm.cdf(-d1)


def up_and_in_put_continuous(S0, K, r, sigma, T, B):
    """Up-and-in put with a continuously monitored barrier B >= K.

    p = -S0 (B/S0)^(2*lam) N(-y) + K e^{-rT} (B/S0)^(2*lam - 2) N(-y + sigma*sqrt(T)),
    lam = (r + sigma^2/2) / sigma^2,
    y   = ln(B^2 / (S0 K)) / (sigma sqrt(T)) + lam * sigma * sqrt(T).
    """
    if B < K:
        raise NotImplementedError("Formula implemented for B >= K only.")
    if S0 >= B:  # already knocked in
        return vanilla_put(S0, K, r, sigma, T)

    sqT = sigma * np.sqrt(T)
    lam = (r + 0.5 * sigma**2) / sigma**2
    y = np.log(B**2 / (S0 * K)) / sqT + lam * sqT
    return (
        -S0 * (B / S0) ** (2 * lam) * norm.cdf(-y)
        + K * np.exp(-r * T) * (B / S0) ** (2 * lam - 2) * norm.cdf(-y + sqT)
    )


def bgk_corrected_price(S0, K, r, sigma, T, B, n_steps, beta=BGK_BETA):
    """Approximate price with n_steps equally spaced monitoring dates.

    The continuous formula is evaluated at the shifted barrier
    B * exp(beta * sigma * sqrt(T / n_steps)); error is o(1/sqrt(n_steps)).
    """
    B_shifted = B * np.exp(beta * sigma * np.sqrt(T / n_steps))
    return up_and_in_put_continuous(S0, K, r, sigma, T, B_shifted)
