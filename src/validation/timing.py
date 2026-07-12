"""Market-timing regressions (Treynor-Mazuy, Henriksson-Merton).

A linear factor regression cannot tell a timing strategy from a static
beta — but timing is exactly what a momentum overlay on an index claims
to do, and the TSMOM literature shows its payoff is *convex* in the
market return (long-straddle-like). The two industry-standard tests add
a convexity term to the CAPM regression and ask whether its coefficient
γ is positive and significant:

    Treynor-Mazuy (1966):       r_s = α + β·r_b + γ·r_b²        + ε
    Henriksson-Merton (1981):   r_s = α + β·r_b + γ·max(r_b, 0) + ε

γ > 0 means the strategy holds more market exposure when the market
rises than when it falls — genuine timing skill (or a convex payoff).
OLS via numpy least squares with classic t-statistics; complements the
linear :func:`src.reporting.attribution.factor_regression`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class TimingTestResult:
    """Output of a market-timing regression.

    Attributes:
        alpha: Per-period intercept.
        alpha_tstat: t-statistic of the intercept.
        beta: Linear market loading.
        gamma: Timing (convexity) coefficient — positive = timing skill.
        gamma_tstat: t-statistic of gamma.
        r_squared: Coefficient of determination.
        n_obs: Observations used.
    """

    alpha: float
    alpha_tstat: float
    beta: float
    gamma: float
    gamma_tstat: float
    r_squared: float
    n_obs: int


def _timing_regression(
    strategy_returns: pd.Series,
    benchmark_returns: pd.Series,
    timing_term: np.ndarray,
) -> TimingTestResult:
    """Shared three-regressor OLS with classic t-statistics."""
    y = strategy_returns.to_numpy(dtype=float)
    rb = benchmark_returns.to_numpy(dtype=float)
    n = len(y)

    design = np.column_stack([np.ones(n), rb, timing_term])
    coef, *_ = np.linalg.lstsq(design, y, rcond=None)
    residuals = y - design @ coef

    dof = n - design.shape[1]
    s_squared = float(residuals @ residuals) / dof
    cov_coef = s_squared * np.linalg.pinv(design.T @ design)
    std_errors = np.sqrt(np.diag(cov_coef))
    with np.errstate(divide="ignore", invalid="ignore"):
        tstats = np.where(std_errors > 0, coef / std_errors, np.inf)

    total = float(((y - y.mean()) ** 2).sum())
    r_squared = 1.0 - float(residuals @ residuals) / total if total > 0 else 0.0

    return TimingTestResult(
        alpha=float(coef[0]),
        alpha_tstat=float(tstats[0]),
        beta=float(coef[1]),
        gamma=float(coef[2]),
        gamma_tstat=float(tstats[2]),
        r_squared=r_squared,
        n_obs=n,
    )


def _validate(strategy_returns: pd.Series, benchmark_returns: pd.Series) -> None:
    if not strategy_returns.index.equals(benchmark_returns.index):
        raise ValueError("strategy and benchmark returns must share the same index.")
    if len(strategy_returns) < 10:
        raise ValueError(f"need at least 10 observations, got {len(strategy_returns)}.")


def treynor_mazuy(
    strategy_returns: pd.Series,
    benchmark_returns: pd.Series,
) -> TimingTestResult:
    """Treynor-Mazuy quadratic timing test (``γ`` on ``r_b²``).

    Args:
        strategy_returns: Per-bar strategy returns.
        benchmark_returns: Per-bar benchmark returns on the same index.

    Returns:
        A :class:`TimingTestResult`; ``gamma > 0`` with a significant
        t-statistic indicates convexity / timing skill.

    Raises:
        ValueError: If the indexes differ or there are < 10 observations.
    """
    _validate(strategy_returns, benchmark_returns)
    rb = benchmark_returns.to_numpy(dtype=float)
    return _timing_regression(strategy_returns, benchmark_returns, rb**2)


def henriksson_merton(
    strategy_returns: pd.Series,
    benchmark_returns: pd.Series,
) -> TimingTestResult:
    """Henriksson-Merton option-style timing test (``γ`` on ``max(r_b, 0)``).

    Args:
        strategy_returns: Per-bar strategy returns.
        benchmark_returns: Per-bar benchmark returns on the same index.

    Returns:
        A :class:`TimingTestResult`; ``gamma`` measures the free call
        option on the market that successful timing synthesises.

    Raises:
        ValueError: If the indexes differ or there are < 10 observations.
    """
    _validate(strategy_returns, benchmark_returns)
    rb = benchmark_returns.to_numpy(dtype=float)
    return _timing_regression(strategy_returns, benchmark_returns, np.maximum(rb, 0.0))
