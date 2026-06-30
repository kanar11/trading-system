"""Advanced risk / performance metrics.

Complements ``src.reporting.metrics`` (Sharpe / Sortino / CAGR /
Calmar / Max Drawdown) with the metrics professional research desks
typically report alongside them:

    * Value-at-Risk (historical & parametric)
    * Conditional VaR / Expected Shortfall (historical)
    * Omega ratio
    * Ulcer Index
    * Gain-to-Pain ratio
    * Drawdown duration / recovery time
    * Rolling beta vs a benchmark
    * Downside / upside deviation
    * Tail ratio
    * Common ratio (CAGR / annualised volatility)

All routines accept a pandas Series of *daily* returns (not
cumulative). They are pure numpy / pandas — no scipy.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import cast

import numpy as np
import pandas as pd

TRADING_DAYS = 252


# ---------------------------------------------------------------------------
# Value-at-Risk family
# ---------------------------------------------------------------------------


def historical_var(returns: pd.Series, level: float = 0.05) -> float:
    """Historical Value-at-Risk at confidence level (1 - ``level``).

    Returns a *positive* loss number — VaR is conventionally quoted as
    the worst loss expected at the given confidence. For ``level=0.05``
    this is the 5% worst-case daily loss.
    """
    if not 0 < level < 1:
        raise ValueError("level must be in (0, 1)")
    r = pd.Series(returns).dropna()
    if r.empty:
        return 0.0
    q = float(np.quantile(r.to_numpy(), level))
    return float(-min(q, 0.0))


def historical_cvar(returns: pd.Series, level: float = 0.05) -> float:
    """Historical Conditional VaR (Expected Shortfall) at ``level``.

    Average loss conditional on the loss being worse than VaR(level).
    Always >= VaR(level).
    """
    if not 0 < level < 1:
        raise ValueError("level must be in (0, 1)")
    r = pd.Series(returns).dropna()
    if r.empty:
        return 0.0
    threshold = float(np.quantile(r.to_numpy(), level))
    tail = r[r <= threshold]
    if tail.empty:
        return 0.0
    return float(-tail.mean())


def parametric_var(returns: pd.Series, level: float = 0.05) -> float:
    """Parametric (Gaussian) VaR at level.

    Assumes daily returns are normal with sample mean / std. Convenient
    closed form, but underestimates tail loss for heavy-tailed series.
    """
    if not 0 < level < 1:
        raise ValueError("level must be in (0, 1)")
    r = pd.Series(returns).dropna()
    if len(r) < 2:
        return 0.0
    mu = float(r.mean())
    sigma = float(r.std(ddof=1))
    # inverse normal at the level — use approximation from stat_tests
    from src.validation.stat_tests import _norm_quantile

    z = _norm_quantile(level)
    return float(-(mu + z * sigma))


# ---------------------------------------------------------------------------
# Tail-shape / drawdown metrics
# ---------------------------------------------------------------------------


def omega_ratio(returns: pd.Series, target_return: float = 0.0) -> float:
    """Omega ratio: gains-above-target / losses-below-target.

    A target of 0 is the "loss-aversion" version. Values > 1 indicate
    more upside than downside relative to the threshold.
    """
    r = pd.Series(returns).dropna()
    if r.empty:
        return 0.0
    excess = r - target_return
    gains = excess[excess > 0].sum()
    losses = -excess[excess < 0].sum()
    if losses == 0:
        return float("inf") if gains > 0 else 0.0
    return float(gains / losses)


def ulcer_index(returns: pd.Series) -> float:
    """Ulcer Index — RMS of percentage drawdowns from running peak.

    Smaller is better. Captures both depth and persistence of drawdowns.
    """
    r = pd.Series(returns).dropna()
    if r.empty:
        return 0.0
    equity = (1 + r).cumprod()
    drawdown_pct = 100 * (equity / equity.cummax() - 1)
    return float(np.sqrt((drawdown_pct**2).mean()))


def gain_to_pain_ratio(returns: pd.Series) -> float:
    """Sum of positive returns divided by sum of absolute negative returns.

    Equivalent to a profit-factor on daily returns. Values > 1 = net
    gainer; > 2 = robustly profitable.
    """
    r = pd.Series(returns).dropna()
    if r.empty:
        return 0.0
    gains = float(r[r > 0].sum())
    losses = float(-r[r < 0].sum())
    if losses == 0:
        return float("inf") if gains > 0 else 0.0
    return gains / losses


@dataclass
class DrawdownStats:
    max_drawdown: float
    max_drawdown_start: pd.Timestamp | None
    max_drawdown_end: pd.Timestamp | None
    recovery_date: pd.Timestamp | None
    duration_days: int
    recovery_days: int | None


def drawdown_stats(returns: pd.Series) -> DrawdownStats:
    """Detailed drawdown analysis: depth, duration, recovery time.

    ``recovery_days`` is ``None`` if the equity curve has not yet
    recovered to its prior peak by the end of the series.
    """
    r = pd.Series(returns).dropna()
    if r.empty:
        return DrawdownStats(0.0, None, None, None, 0, None)
    equity = (1 + r).cumprod()
    peak = equity.cummax()
    dd = equity / peak - 1
    if dd.min() >= 0:
        return DrawdownStats(0.0, None, None, None, 0, None)

    end_idx = cast("pd.Timestamp", dd.idxmin())
    max_dd = float(dd.loc[end_idx])

    # start of drawdown = last bar at a peak before end_idx
    peak_value = peak.loc[end_idx]
    pre = equity.loc[:end_idx]
    start_idx = pre[pre == peak_value].index[0]

    # recovery = first bar AFTER end_idx where equity >= prior peak
    post = equity.loc[end_idx:]
    recovered = post[post >= peak_value]
    recovery_idx = recovered.index[0] if not recovered.empty else None
    recovery_days = (recovery_idx - end_idx).days if recovery_idx is not None else None

    duration_days = (end_idx - start_idx).days

    return DrawdownStats(
        max_drawdown=max_dd,
        max_drawdown_start=start_idx,
        max_drawdown_end=end_idx,
        recovery_date=recovery_idx,
        duration_days=duration_days,
        recovery_days=recovery_days,
    )


# ---------------------------------------------------------------------------
# Cross-sectional & deviation metrics
# ---------------------------------------------------------------------------


def downside_deviation(returns: pd.Series, target: float = 0.0) -> float:
    """Standard deviation of returns below the target. Annualised."""
    r = pd.Series(returns).dropna()
    below = r[r < target]
    if below.empty:
        return 0.0
    return float(below.std(ddof=1) * math.sqrt(TRADING_DAYS))


def upside_deviation(returns: pd.Series, target: float = 0.0) -> float:
    """Standard deviation of returns above the target. Annualised."""
    r = pd.Series(returns).dropna()
    above = r[r > target]
    if above.empty:
        return 0.0
    return float(above.std(ddof=1) * math.sqrt(TRADING_DAYS))


def tail_ratio(returns: pd.Series, level: float = 0.05) -> float:
    """``|quantile(returns, 1 - level)| / |quantile(returns, level)|``.

    Values > 1 indicate fat right tail relative to left tail.
    """
    r = pd.Series(returns).dropna()
    if r.empty:
        return 0.0
    right = abs(float(np.quantile(r.to_numpy(), 1 - level)))
    left = abs(float(np.quantile(r.to_numpy(), level)))
    if left == 0:
        return float("inf") if right > 0 else 0.0
    return right / left


def common_ratio(returns: pd.Series) -> float:
    """CAGR divided by annualised volatility.

    Equivalent to an unannualised Sharpe with zero risk-free rate but
    using compound returns in the numerator instead of arithmetic mean.
    """
    r = pd.Series(returns).dropna()
    if r.empty:
        return 0.0
    equity = (1 + r).cumprod()
    years = len(r) / TRADING_DAYS
    cagr = equity.iloc[-1] ** (1 / years) - 1 if years > 0 else 0.0
    vol = float(r.std(ddof=1) * math.sqrt(TRADING_DAYS))
    if vol == 0:
        return 0.0
    return float(cagr / vol)


def rolling_beta(
    strategy_returns: pd.Series,
    benchmark_returns: pd.Series,
    window: int = 60,
) -> pd.Series:
    """Rolling OLS beta of strategy on benchmark.

    Args:
        strategy_returns: Daily strategy returns.
        benchmark_returns: Daily benchmark returns.
        window: Rolling window in days.

    Returns:
        Series of betas indexed by ``strategy_returns.index``. First
        ``window-1`` values are NaN.
    """
    df = pd.concat(
        [strategy_returns.rename("s"), benchmark_returns.rename("b")],
        axis=1,
        join="inner",
    )
    cov = df["s"].rolling(window).cov(df["b"])
    var = df["b"].rolling(window).var()
    return (cov / var.replace(0, np.nan)).rename("beta")


# ---------------------------------------------------------------------------
# Distribution shape & benchmark-relative / drawdown-adjusted ratios
# ---------------------------------------------------------------------------


def _annualized_return(r: pd.Series) -> float:
    """Geometric annualised return of a daily return series."""
    years = len(r) / TRADING_DAYS
    if years <= 0:
        return 0.0
    total_growth = float((1.0 + r).to_numpy().prod())
    if total_growth <= 0:
        return -1.0
    return float(total_growth ** (1 / years) - 1)


def skewness(returns: pd.Series) -> float:
    """Sample skewness of the return distribution (0 = symmetric)."""
    r = pd.Series(returns).dropna()
    if len(r) < 3:
        return 0.0
    mean = float(r.mean())
    std = float(r.std(ddof=0))
    if std == 0:
        return 0.0
    return float((((r - mean) / std) ** 3).mean())


def kurtosis(returns: pd.Series) -> float:
    """Excess kurtosis (0 = Gaussian; > 0 = heavy tails / fat outliers)."""
    r = pd.Series(returns).dropna()
    if len(r) < 4:
        return 0.0
    mean = float(r.mean())
    std = float(r.std(ddof=0))
    if std == 0:
        return 0.0
    return float((((r - mean) / std) ** 4).mean() - 3.0)


def tracking_error(
    returns: pd.Series,
    benchmark: pd.Series,
    periods_per_year: int = TRADING_DAYS,
) -> float:
    """Annualised standard deviation of the active (returns - benchmark) series."""
    df = pd.concat([returns.rename("r"), benchmark.rename("b")], axis=1, join="inner").dropna()
    if len(df) < 2:
        return 0.0
    active = df["r"] - df["b"]
    return float(active.std(ddof=1) * math.sqrt(periods_per_year))


def information_ratio(
    returns: pd.Series,
    benchmark: pd.Series,
    periods_per_year: int = TRADING_DAYS,
) -> float:
    """Annualised active return divided by tracking error.

    Measures risk-adjusted out-performance versus a benchmark. Returns 0 when
    the active series has no variability (degenerate tracking error).
    """
    df = pd.concat([returns.rename("r"), benchmark.rename("b")], axis=1, join="inner").dropna()
    if len(df) < 2:
        return 0.0
    active = df["r"] - df["b"]
    te = float(active.std(ddof=1))
    if te == 0:
        return 0.0
    return float(active.mean() / te * math.sqrt(periods_per_year))


def sterling_ratio(returns: pd.Series) -> float:
    """Annualised return divided by the average drawdown magnitude.

    Returns +inf when the equity curve never draws down but is profitable.
    """
    r = pd.Series(returns).dropna()
    if r.empty:
        return 0.0
    equity = (1 + r).cumprod()
    dd = equity / equity.cummax() - 1
    avg_dd = float(-dd[dd < 0].mean()) if bool((dd < 0).any()) else 0.0
    ann = _annualized_return(r)
    if avg_dd == 0:
        return float("inf") if ann > 0 else 0.0
    return float(ann / avg_dd)


def burke_ratio(returns: pd.Series) -> float:
    """Annualised return divided by the root-sum-square of drawdowns.

    Penalises a few deep drawdowns more than many shallow ones. Returns +inf
    when the equity curve never draws down but is profitable.
    """
    r = pd.Series(returns).dropna()
    if r.empty:
        return 0.0
    equity = (1 + r).cumprod()
    dd = equity / equity.cummax() - 1
    rss = float(np.sqrt((dd**2).sum()))
    ann = _annualized_return(r)
    if rss == 0:
        return float("inf") if ann > 0 else 0.0
    return float(ann / rss)


# ---------------------------------------------------------------------------
# CAPM / benchmark-relative ratios
# ---------------------------------------------------------------------------


def _capm_inputs(
    returns: pd.Series, benchmark: pd.Series
) -> tuple[pd.Series, pd.Series, float] | None:
    """Align returns & benchmark and return (strat, bench, beta), or None."""
    df = pd.concat([returns.rename("s"), benchmark.rename("b")], axis=1, join="inner").dropna()
    if len(df) < 2:
        return None
    strat, bench = df["s"], df["b"]
    var_b = float(bench.var(ddof=1))
    if var_b == 0:
        return None
    beta = float(strat.cov(bench)) / var_b
    return strat, bench, beta


def treynor_ratio(returns: pd.Series, benchmark: pd.Series, rf_daily: float = 0.0) -> float:
    """Annualised excess return per unit of market beta (CAPM systematic risk).

    Returns 0.0 when beta is zero or there is too little overlapping data.
    """
    inputs = _capm_inputs(returns, benchmark)
    if inputs is None:
        return 0.0
    strat, _, beta = inputs
    if beta == 0:
        return 0.0
    return float((float(strat.mean()) - rf_daily) * TRADING_DAYS / beta)


def jensen_alpha(returns: pd.Series, benchmark: pd.Series, rf_daily: float = 0.0) -> float:
    """Annualised CAPM alpha: actual excess return minus beta-predicted excess.

    ``alpha = (r - rf) - beta * (r_bench - rf)``, annualised. Positive alpha is
    out-performance not explained by market exposure.
    """
    inputs = _capm_inputs(returns, benchmark)
    if inputs is None:
        return 0.0
    strat, bench, beta = inputs
    alpha_daily = (float(strat.mean()) - rf_daily) - beta * (float(bench.mean()) - rf_daily)
    return float(alpha_daily * TRADING_DAYS)


def m2_ratio(returns: pd.Series, benchmark: pd.Series, rf_daily: float = 0.0) -> float:
    """Modigliani M²: the strategy's return re-levered to the benchmark's volatility.

    ``M2 = rf + Sharpe_strategy * sigma_benchmark`` (annualised), expressing
    risk-adjusted performance in the same units as the benchmark's return.
    Returns 0.0 when the strategy has zero volatility or too little data.
    """
    inputs = _capm_inputs(returns, benchmark)
    if inputs is None:
        return 0.0
    strat, bench, _ = inputs
    strat_std = float(strat.std(ddof=1))
    if strat_std == 0:
        return 0.0
    sharpe_daily = (float(strat.mean()) - rf_daily) / strat_std
    m2_daily = rf_daily + sharpe_daily * float(bench.std(ddof=1))
    return float(m2_daily * TRADING_DAYS)
