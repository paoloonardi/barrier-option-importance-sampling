import numpy as np
import pytest

from barrier_is import (
    bgk_corrected_price,
    optimal_tilts,
    price_importance_sampling,
    price_standard_mc,
    up_and_in_put_continuous,
    vanilla_put,
)

P = dict(S0=100.0, K=100.0, r=0.05, sigma=0.2, T=1.0, B=120.0)
N = 252


def test_continuous_price_reference_value():
    assert up_and_in_put_continuous(**P) == pytest.approx(0.21340, abs=1e-4)


def test_barrier_at_spot_equals_vanilla_put():
    p = dict(P, B=P["S0"])
    expected = vanilla_put(P["S0"], P["K"], P["r"], P["sigma"], P["T"])
    assert up_and_in_put_continuous(**p) == pytest.approx(expected, rel=1e-10)


def test_far_barrier_is_worthless():
    assert up_and_in_put_continuous(**dict(P, B=1e4)) < 1e-12


def test_bgk_lowers_price_for_up_barrier():
    assert bgk_corrected_price(**P, n_steps=N) < up_and_in_put_continuous(**P)


def test_tilt_cumulants_match():
    """psi(theta_plus) == psi(theta_minus): justifies the n * psi term."""
    dt = P["T"] / N
    mu = (P["r"] - 0.5 * P["sigma"] ** 2) * dt
    psi = lambda th: mu * th + 0.5 * P["sigma"] ** 2 * dt * th**2
    tp, tm = optimal_tilts(**P)
    assert psi(tp) == pytest.approx(psi(tm), rel=1e-12)


def _reference_loop(Z, S0, K, r, sigma, T, B, n):
    """Straightforward per-path loop (as in the original course script)."""
    dt = T / n
    mu = (r - 0.5 * sigma**2) * dt
    vol = sigma * np.sqrt(dt)
    b = np.log(B / S0)
    tp, tm = optimal_tilts(S0, K, r, sigma, T, B)
    psi = mu * tm + 0.5 * sigma**2 * dt * tm**2
    out = np.zeros(len(Z))
    for i, z in enumerate(Z):
        x, tau, x_tau = 0.0, None, 0.0
        for t in range(n):
            th = tp if tau is None else tm
            x += mu + sigma**2 * dt * th + vol * z[t]
            if tau is None and x >= b:
                tau, x_tau = t, x
        if tau is not None:
            lr = np.exp((tm - tp) * x_tau - tm * x + n * psi)
            out[i] = np.exp(-r * T) * max(K - S0 * np.exp(x), 0.0) * lr
    return out


def test_vectorised_is_matches_loop():
    m, seed = 200, 123
    Z = np.random.default_rng(seed).standard_normal((m, N))
    expected = _reference_loop(Z, **P, n=N).mean()
    got = price_importance_sampling(**P, n_steps=N, n_paths=m, seed=seed).price
    assert got == pytest.approx(expected, rel=1e-10)


def test_is_unbiased_and_lower_variance():
    mc = price_standard_mc(**P, n_steps=N, n_paths=40_000, seed=1)
    is_ = price_importance_sampling(**P, n_steps=N, n_paths=40_000, seed=2)
    pooled_se = np.hypot(mc.std_error, is_.std_error)
    assert abs(mc.price - is_.price) < 4 * pooled_se
    assert is_.std_error < mc.std_error / 3
