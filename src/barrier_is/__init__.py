"""Pricing of discretely monitored up-and-in put options with Monte Carlo
and importance sampling."""

from .analytical import (
    BGK_BETA,
    bgk_corrected_price,
    up_and_in_put_continuous,
    vanilla_put,
)
from .monte_carlo import MCResult, price_standard_mc, simulate_gbm_paths
from .importance_sampling import optimal_tilts, price_importance_sampling

__all__ = [
    "BGK_BETA",
    "bgk_corrected_price",
    "up_and_in_put_continuous",
    "vanilla_put",
    "MCResult",
    "price_standard_mc",
    "simulate_gbm_paths",
    "optimal_tilts",
    "price_importance_sampling",
]
__version__ = "1.0.0"
