"""Mean-variance efficient frontier.

Traces the Markowitz efficient frontier as a grid of minimum-variance
portfolios, one per target return between the lowest and highest asset mean.
Each point is the closed-form solution of

    min wᵀΣw   s.t.   wᵀ1 = 1,  wᵀμ = m

(Merton, 1972): with ``A = 1ᵀΣ⁻¹1``, ``B = 1ᵀΣ⁻¹μ``, ``C = μᵀΣ⁻¹μ`` and
``D = AC − B²``,

    w(m) = ((C − Bm)/D) Σ⁻¹1 + ((Am − B)/D) Σ⁻¹μ.

The closed form allows shorts; by default (``allow_short=False``) negative
weights are clipped and re-normalised, matching the convention of
:mod:`src.portfolio.optimizer` (an approximation of the true long-only
frontier — exact solutions would need a QP solver). Realised return and
volatility are always recomputed from the final weights, so the reported
points are attainable either way. Pure numpy, no scipy.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.portfolio.optimizer import _clip_and_normalise, _covariance_matrix

logger = logging.getLogger(__name__)


@dataclass
class EfficientFrontier:
    """A sampled efficient frontier.

    Attributes:
        expected_returns: Per-point portfolio mean return (same periodicity
            as the input returns).
        volatilities: Per-point portfolio standard deviation.
        sharpe_ratios: Per-point ``(return − rf) / volatility`` (0 where the
            volatility is 0).
        weights: Per-point weight vectors — rows are frontier points, columns
            are tickers; every row sums to 1.
    """

    expected_returns: np.ndarray
    volatilities: np.ndarray
    sharpe_ratios: np.ndarray
    weights: pd.DataFrame

    @property
    def min_volatility_index(self) -> int:
        """Index of the lowest-volatility frontier point."""
        return int(np.argmin(self.volatilities))

    @property
    def max_sharpe_index(self) -> int:
        """Index of the highest-Sharpe frontier point."""
        return int(np.argmax(self.sharpe_ratios))


def efficient_frontier(
    returns: pd.DataFrame,
    n_points: int = 20,
    cov: pd.DataFrame | np.ndarray | None = None,
    rf_daily: float = 0.0,
    allow_short: bool = False,
) -> EfficientFrontier:
    """Sample ``n_points`` portfolios along the mean-variance frontier.

    Target returns are spaced evenly between the smallest and the largest
    asset mean return (the range attainable by long-only combinations).

    Args:
        returns: DataFrame of asset returns (columns = tickers).
        n_points: Number of frontier points (>= 2).
        cov: Optional pre-computed covariance matrix.
        rf_daily: Risk-free rate per period, used for the Sharpe ratios.
        allow_short: If True, keep the exact (possibly short) closed-form
            weights; if False (default), clip negatives and re-normalise.

    Returns:
        An :class:`EfficientFrontier` with ``n_points`` points.

    Raises:
        ValueError: If ``n_points < 2`` or ``returns`` has no columns.
    """
    if n_points < 2:
        raise ValueError(f"n_points must be >= 2, got {n_points}.")
    if returns.shape[1] == 0:
        raise ValueError("returns must have at least one column.")

    cov_matrix = _covariance_matrix(returns, cov)
    mu = returns.mean().to_numpy(dtype=float)

    try:
        inv = np.linalg.inv(cov_matrix)
    except np.linalg.LinAlgError:
        logger.warning("covariance not invertible; using pseudo-inverse")
        inv = np.linalg.pinv(cov_matrix)

    ones = np.ones(cov_matrix.shape[0])
    inv_ones = inv @ ones
    inv_mu = inv @ mu
    a = float(ones @ inv_ones)
    b = float(ones @ inv_mu)
    c = float(mu @ inv_mu)
    d = a * c - b * b

    targets = np.linspace(float(mu.min()), float(mu.max()), n_points)
    degenerate = not np.isfinite(d) or abs(d) < 1e-16
    if degenerate:
        # all asset means (effectively) equal — the frontier collapses to the
        # global minimum-variance portfolio
        logger.warning("degenerate frontier (equal means); returning min-variance point")

    weight_rows = np.empty((n_points, cov_matrix.shape[0]))
    for i, target in enumerate(targets):
        if degenerate:
            w = inv_ones / a
        else:
            gamma = (c - b * target) / d
            lam = (a * target - b) / d
            w = gamma * inv_ones + lam * inv_mu
        if not allow_short:
            w = _clip_and_normalise(w)
        weight_rows[i] = w

    expected = weight_rows @ mu
    variances = np.einsum("ij,jk,ik->i", weight_rows, cov_matrix, weight_rows)
    vols = np.sqrt(np.maximum(variances, 0.0))
    with np.errstate(divide="ignore", invalid="ignore"):
        sharpes = np.where(vols > 0, (expected - rf_daily) / vols, 0.0)

    return EfficientFrontier(
        expected_returns=expected,
        volatilities=vols,
        sharpe_ratios=sharpes,
        weights=pd.DataFrame(weight_rows, columns=returns.columns),
    )
