"""Covariance-aware portfolio weight optimisation.

Extends ``src.portfolio.portfolio`` (which only knows equal / inverse-vol
weights) with three correlation-aware allocation schemes:

    * ``min_variance_weights``     — closed-form, long-only by default.
    * ``max_sharpe_weights``       — closed-form tangency portfolio.
    * ``risk_parity_weights``      — iterative equal risk-contribution.

All routines accept either a returns DataFrame (covariance is estimated
internally with a small ridge for numerical stability) or a pre-computed
covariance matrix. They never short — negative weights from the
closed-form solutions are clipped and the result re-normalised. For
strictly long-only optimal solutions you'd want a QP solver, but in
practice this clip-and-renormalise is a good approximation for
typical multi-asset baskets.

No scipy required — pure numpy.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _covariance_matrix(
    returns: pd.DataFrame,
    cov: pd.DataFrame | np.ndarray | None,
    ridge: float = 1e-8,
) -> np.ndarray:
    """Resolve ``cov`` (caller-supplied or estimated from ``returns``) and
    return it as a numpy array with a small diagonal ridge for stability."""
    if cov is None:
        cov_arr = returns.cov().values
    elif isinstance(cov, pd.DataFrame):
        cov_arr = cov.values
    else:
        cov_arr = np.asarray(cov, dtype=float)

    n_cov = cov_arr.shape[0]
    if cov_arr.ndim != 2 or cov_arr.shape != (n_cov, n_cov):
        raise ValueError(f"covariance must be square, got shape {cov_arr.shape}")
    if cov is not None and n_cov != returns.shape[1]:
        raise ValueError(
            f"covariance shape {cov_arr.shape} does not match "
            f"returns with {returns.shape[1]} columns"
        )
    return cov_arr + ridge * np.eye(n_cov)


def _clip_and_normalise(weights: np.ndarray) -> np.ndarray:
    """Clip negative weights to zero and re-normalise to sum to 1.

    Falls back to equal weights if the entire vector is non-positive.
    """
    clipped = np.maximum(weights, 0.0)
    total = clipped.sum()
    if total <= 0:
        return np.full_like(weights, 1.0 / len(weights))
    return clipped / total


def min_variance_weights(
    returns: pd.DataFrame,
    cov: pd.DataFrame | np.ndarray | None = None,
) -> pd.Series:
    """Long-only minimum-variance portfolio weights.

    Closed form (unconstrained):
        w = Σ⁻¹ 1 / (1ᵀ Σ⁻¹ 1)

    Negative weights are clipped to zero and the vector is re-normalised.

    Args:
        returns: DataFrame of asset returns (columns = tickers).
        cov: Optional pre-computed covariance matrix.

    Returns:
        Series of weights indexed by ticker, summing to 1.
    """
    Σ = _covariance_matrix(returns, cov)
    ones = np.ones(Σ.shape[0])
    try:
        inv = np.linalg.inv(Σ)
    except np.linalg.LinAlgError:
        logger.warning("covariance not invertible; falling back to equal weights")
        n = Σ.shape[0]
        return pd.Series(np.full(n, 1.0 / n), index=returns.columns)

    raw = inv @ ones
    w = _clip_and_normalise(raw / (ones @ raw))
    return pd.Series(w, index=returns.columns, name="min_variance")


def max_sharpe_weights(
    returns: pd.DataFrame,
    cov: pd.DataFrame | np.ndarray | None = None,
    rf_daily: float = 0.0,
) -> pd.Series:
    """Long-only tangency (max Sharpe) portfolio weights.

    Closed form (unconstrained):
        w = Σ⁻¹ (μ - rf) / (1ᵀ Σ⁻¹ (μ - rf))

    Negative weights are clipped to zero and the vector is re-normalised.

    Args:
        returns: DataFrame of asset returns (columns = tickers).
        cov: Optional pre-computed covariance matrix.
        rf_daily: Daily risk-free rate.

    Returns:
        Series of weights indexed by ticker, summing to 1.
    """
    Σ = _covariance_matrix(returns, cov)
    excess = returns.mean().values - rf_daily

    try:
        inv = np.linalg.inv(Σ)
    except np.linalg.LinAlgError:
        logger.warning("covariance not invertible; falling back to equal weights")
        n = Σ.shape[0]
        return pd.Series(np.full(n, 1.0 / n), index=returns.columns)

    raw = inv @ excess
    denom = float(np.ones(Σ.shape[0]) @ raw)
    if denom == 0 or not np.isfinite(denom):
        # zero net excess return — fall back to min-variance
        return min_variance_weights(returns, cov).rename("max_sharpe")

    w = _clip_and_normalise(raw / denom)
    return pd.Series(w, index=returns.columns, name="max_sharpe")


def risk_parity_weights(
    returns: pd.DataFrame,
    cov: pd.DataFrame | np.ndarray | None = None,
    max_iter: int = 500,
    tol: float = 1e-8,
) -> pd.Series:
    """Equal risk-contribution (risk-parity) portfolio weights.

    Solves for w such that each asset contributes the same amount to
    total portfolio variance: w_i * (Σw)_i = constant for all i.

    Uses the cyclical coordinate-descent algorithm of Maillard,
    Roncalli & Teïletche (2010): at each step, update one weight
    analytically while holding the others fixed by solving the
    scalar quadratic

        Σ_ii · w_i² + m_i · w_i − b_i = 0,
        m_i = Σ_{j≠i} Σ_ij · w_j,

    which is the partial KKT condition for the unconstrained
    log-barrier objective ½ wᵀ Σ w − Σ_i b_i log w_i. This update
    is known to converge globally for positive-definite Σ. Weights
    are renormalised to sum to 1 at the end.

    Args:
        returns: DataFrame of asset returns.
        cov: Optional pre-computed covariance matrix.
        max_iter: Iteration cap.
        tol: Convergence tolerance on the weight L1 change.

    Returns:
        Series of weights indexed by ticker, summing to 1.
    """
    Σ = _covariance_matrix(returns, cov)
    n = Σ.shape[0]
    b = np.full(n, 1.0 / n)  # equal risk budget per asset
    w = np.full(n, 1.0 / n)

    for _ in range(max_iter):
        w_prev = w.copy()
        for i in range(n):
            a = float(Σ[i, i])
            m = float(Σ[i] @ w) - a * w[i]
            # solve a w_i² + m w_i - b_i = 0, positive root
            disc = m * m + 4 * a * b[i]
            w[i] = (-m + float(np.sqrt(disc))) / (2 * a)
        if np.abs(w - w_prev).sum() < tol:
            break

    w = w / w.sum()
    return pd.Series(w, index=returns.columns, name="risk_parity")
