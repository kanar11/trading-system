"""EWMA (RiskMetrics) covariance estimation.

The sample covariance weights a bar from three years ago the same as
yesterday's; after a volatility regime shift it stays wrong for months.
The RiskMetrics answer is exponential weighting::

    Σ_t = λ Σ_{t-1} + (1 − λ) x_t x_tᵀ

equivalently a Gram matrix under weights ``w_i ∝ (1−λ)·λ^age`` (here
normalised to sum to one, the finite-sample form). With the classic
``λ = 0.94`` for daily data the estimate tracks current conditions; as
``λ → 1`` it converges to the plain sample covariance.

The result is a labelled DataFrame that drops into the ``cov=`` argument
of every optimiser in this package — the *responsive* third member of
the estimator family alongside the sample matrix and the Ledoit-Wolf
shrinkage (#53). Positive semi-definite by construction (a weighted
Gram matrix).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def ewma_covariance(
    returns: pd.DataFrame,
    decay: float = 0.94,
    demean: bool = False,
) -> pd.DataFrame:
    """Exponentially weighted covariance of a return panel.

    Args:
        returns: DataFrame of asset returns (rows = observations, columns
            = tickers), at least 2 rows, NaN-free.
        decay: RiskMetrics λ in (0, 1); 0.94 is the classic daily value,
            larger = longer memory.
        demean: If True subtract the (equally weighted) sample mean first;
            RiskMetrics convention is zero-mean, hence False by default.

    Returns:
        Covariance DataFrame (tickers × tickers).

    Raises:
        ValueError: If ``decay`` is outside (0, 1), the panel has fewer
            than 2 rows/no columns, or contains NaNs.
    """
    if not 0.0 < decay < 1.0:
        raise ValueError(f"decay must be in (0, 1), got {decay}.")
    if returns.shape[1] == 0:
        raise ValueError("returns must have at least one column.")
    if returns.shape[0] < 2:
        raise ValueError(f"returns must have at least 2 rows, got {returns.shape[0]}.")

    x = returns.to_numpy(dtype=float)
    if np.isnan(x).any():
        raise ValueError("returns must not contain NaNs.")
    if demean:
        x = x - x.mean(axis=0)

    n = x.shape[0]
    ages = np.arange(n - 1, -1, -1, dtype=float)  # newest row has age 0
    weights = decay**ages
    weights /= weights.sum()

    cov = (x * weights[:, np.newaxis]).T @ x
    return pd.DataFrame(cov, index=returns.columns, columns=returns.columns)
