"""Portfolio risk / diversification analytics.

Read-only diagnostics for a set of portfolio weights and a covariance matrix:
total volatility, per-asset risk contributions (the basis of risk parity), the
diversification ratio, and an effective-number-of-assets concentration measure.
Pure numpy; complements the weight optimisers in
:mod:`src.portfolio.optimizer`.
"""

from __future__ import annotations

import numpy as np

ArrayLike = np.ndarray | list[float]


def _as_weights_cov(weights: ArrayLike, cov: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    """Coerce inputs to numpy and validate that the covariance matches."""
    w = np.asarray(weights, dtype=float)
    c = np.asarray(cov, dtype=float)
    n = w.shape[0]
    if c.shape != (n, n):
        raise ValueError(f"covariance shape {c.shape} does not match {n} weights")
    return w, c


def portfolio_volatility(weights: ArrayLike, cov: ArrayLike) -> float:
    """Portfolio standard deviation ``sqrt(wᵀ Σ w)``."""
    w, c = _as_weights_cov(weights, cov)
    return float(np.sqrt(max(float(w @ c @ w), 0.0)))


def risk_contributions(weights: ArrayLike, cov: ArrayLike) -> np.ndarray:
    """Each asset's fractional contribution to portfolio variance.

    ``RC_i = w_i (Σw)_i / (wᵀ Σ w)``; contributions sum to 1 (all-zero when the
    portfolio variance is zero). Equal contributions characterise a risk-parity
    portfolio.
    """
    w, c = _as_weights_cov(weights, cov)
    contrib = w * (c @ w)
    total = float(contrib.sum())
    if total == 0:
        return np.zeros_like(w)
    out: np.ndarray = contrib / total
    return out


def diversification_ratio(weights: ArrayLike, cov: ArrayLike) -> float:
    """Weighted-average asset vol divided by portfolio vol (>= 1).

    ``DR = (w·σ) / sqrt(wᵀ Σ w)``. Equals 1 for a single asset (or perfectly
    correlated holdings) and rises as diversification reduces portfolio vol.
    Returns 0.0 when portfolio volatility is zero.
    """
    w, c = _as_weights_cov(weights, cov)
    weighted_avg_vol = float(w @ np.sqrt(np.diag(c)))
    port_vol = float(np.sqrt(max(float(w @ c @ w), 0.0)))
    if port_vol == 0:
        return 0.0
    return weighted_avg_vol / port_vol


def effective_number_of_assets(weights: ArrayLike) -> float:
    """Inverse Herfindahl of the weights: ``1 / Σ w_i²``.

    The "effective number of bets" — equals N for N equal weights and 1 for a
    fully concentrated portfolio. Returns 0.0 for all-zero weights.
    """
    w = np.asarray(weights, dtype=float)
    sum_sq = float(np.sum(w**2))
    if sum_sq == 0:
        return 0.0
    return 1.0 / sum_sq
