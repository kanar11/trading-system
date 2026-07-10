"""Factor / performance attribution.

Answers the question: "Is my strategy's return driven by genuine
alpha, or by passive exposure to a known factor?" by regressing the
daily strategy returns against one or more factor return streams:

    r_strategy_t = alpha + sum_i beta_i * f_{i,t} + epsilon_t

The intercept (``alpha``) is the annualised excess return that
*cannot* be explained by the factors. Each ``beta`` is the factor
loading. Residual t-stats and R² indicate how well the factor model
explains the return stream.

Implementation uses ordinary least squares via ``numpy.linalg.lstsq``
— no scipy dependency. Suitable for the small N (a few factors,
hundreds-thousands of daily observations) that this project deals with.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

TRADING_DAYS = 252


@dataclass
class AttributionResult:
    """Output of a factor regression.

    Attributes:
        alpha_daily: Intercept (per-day excess return).
        alpha_annualised: ``alpha_daily * 252`` for easier interpretation.
        alpha_tstat: t-statistic for the intercept.
        betas: Dict mapping factor name → loading.
        beta_tstats: Dict mapping factor name → t-statistic.
        r_squared: Coefficient of determination.
        adj_r_squared: R² adjusted for number of factors.
        n_obs: Number of observations used in the regression.
        residuals: Per-observation residual series.
    """

    alpha_daily: float
    alpha_annualised: float
    alpha_tstat: float
    betas: dict[str, float]
    beta_tstats: dict[str, float]
    r_squared: float
    adj_r_squared: float
    n_obs: int
    residuals: pd.Series


def compute_beta(
    strategy_returns: pd.Series,
    benchmark_returns: pd.Series,
) -> float:
    """Simple OLS beta (single-factor) of strategy vs benchmark.

    Args:
        strategy_returns: Daily strategy returns.
        benchmark_returns: Daily benchmark returns (e.g. SPY).

    Returns:
        Beta. Returns 0.0 if the benchmark variance is zero.
    """
    joined = pd.concat([strategy_returns, benchmark_returns], axis=1, join="inner").dropna()
    if len(joined) < 2:
        return 0.0

    s = joined.iloc[:, 0].to_numpy()
    b = joined.iloc[:, 1].to_numpy()

    var = float(np.var(b, ddof=1))
    # tolerance handles "essentially constant" benchmarks where var is
    # a floating-point dust like 2e-37 instead of an exact zero
    if var < 1e-20:
        return 0.0
    cov = float(np.cov(s, b, ddof=1)[0, 1])
    return cov / var


def factor_regression(
    strategy_returns: pd.Series,
    factors: pd.DataFrame,
    rf_rate: pd.Series | float = 0.0,
) -> AttributionResult:
    """Run a multi-factor OLS regression and report alpha + betas with t-stats.

    Args:
        strategy_returns: Daily strategy returns (not cumulative).
        factors: DataFrame of daily factor returns. One column per factor;
            column names become the keys of ``betas`` / ``beta_tstats``.
        rf_rate: Daily risk-free rate, as a Series aligned to the dates or
            a scalar (default 0). The strategy's excess return is regressed.

    Returns:
        An :class:`AttributionResult` summarising the fit.

    Raises:
        ValueError: If there are fewer observations than (factors + 1).
    """
    # align
    df = pd.concat([strategy_returns.rename("y"), factors], axis=1, join="inner").dropna()
    if isinstance(rf_rate, pd.Series):
        df = df.join(rf_rate.rename("rf"), how="inner").dropna()
        df["y"] = df["y"] - df["rf"]
        df = df.drop(columns=["rf"])
    else:
        df["y"] = df["y"] - float(rf_rate)

    n = len(df)
    factor_cols = list(factors.columns)
    k = len(factor_cols)
    if n < k + 2:
        raise ValueError(f"Need at least {k + 2} observations for {k}-factor regression, got {n}.")

    # design matrix with intercept column
    X = np.column_stack([np.ones(n), df[factor_cols].to_numpy()])
    y = df["y"].to_numpy()

    # OLS via lstsq — numerically stable
    coefs, *_ = np.linalg.lstsq(X, y, rcond=None)
    y_hat = X @ coefs
    residuals = y - y_hat

    # statistics
    rss = float((residuals**2).sum())
    tss = float(((y - y.mean()) ** 2).sum())
    r_squared = 1 - rss / tss if tss > 0 else 0.0
    p = X.shape[1]  # 1 + k
    adj_r_squared = 1 - (1 - r_squared) * (n - 1) / max(n - p, 1)

    # standard errors: sigma^2 * (X'X)^-1
    dof = max(n - p, 1)
    sigma2 = rss / dof
    try:
        xtx_inv = np.linalg.inv(X.T @ X)
        se = np.sqrt(np.diag(sigma2 * xtx_inv))
    except np.linalg.LinAlgError:
        logger.warning("(X'X) is singular - t-stats set to nan")
        se = np.full(p, np.nan)

    tstats = np.where(se > 0, coefs / se, np.nan)

    alpha_daily = float(coefs[0])
    return AttributionResult(
        alpha_daily=alpha_daily,
        alpha_annualised=alpha_daily * TRADING_DAYS,
        alpha_tstat=float(tstats[0]),
        betas={name: float(coefs[i + 1]) for i, name in enumerate(factor_cols)},
        beta_tstats={name: float(tstats[i + 1]) for i, name in enumerate(factor_cols)},
        r_squared=float(r_squared),
        adj_r_squared=float(adj_r_squared),
        n_obs=n,
        residuals=pd.Series(residuals, index=df.index, name="residual"),
    )


def print_attribution_report(result: AttributionResult, title: str = "Factor Attribution") -> None:
    """Pretty-print an :class:`AttributionResult`."""
    print("\n" + "=" * 60)
    print(f"{title}  (n = {result.n_obs})")
    print("=" * 60)
    print(f"  Alpha (annualised):  {result.alpha_annualised:>+8.2%}  (t={result.alpha_tstat:+.2f})")
    for name, beta in result.betas.items():
        t = result.beta_tstats[name]
        print(f"  Beta - {name:<12} {beta:>+8.3f}  (t={t:+.2f})")
    print(f"  R^2:                 {result.r_squared:>8.3f}")
    print(f"  Adj. R^2:            {result.adj_r_squared:>8.3f}")
    print("=" * 60)


def _aligned(strategy_returns: pd.Series, benchmark_returns: pd.Series) -> pd.DataFrame:
    """Inner-join and drop NaNs; columns 0 = strategy, 1 = benchmark."""
    return pd.concat(
        [strategy_returns.rename("s"), benchmark_returns.rename("b")],
        axis=1,
        join="inner",
    ).dropna()


def up_capture(strategy_returns: pd.Series, benchmark_returns: pd.Series) -> float:
    """Up-market capture ratio.

    Strategy's mean return on days the benchmark is up, divided by the
    benchmark's mean up return. > 1 means the strategy amplifies the upside.
    Returns 0.0 when there are no up days or no overlapping observations.
    """
    joined = _aligned(strategy_returns, benchmark_returns)
    if joined.empty:
        return 0.0
    strat, bench = joined["s"], joined["b"]
    up = bench > 0
    if not bool(up.any()):
        return 0.0
    bench_up = float(bench[up].mean())
    if bench_up == 0:
        return 0.0
    return float(strat[up].mean()) / bench_up


def down_capture(strategy_returns: pd.Series, benchmark_returns: pd.Series) -> float:
    """Down-market capture ratio.

    Strategy's mean return on days the benchmark is down, divided by the
    benchmark's mean down return. < 1 means the strategy loses less than the
    benchmark in down markets. Returns 0.0 when there are no down days.
    """
    joined = _aligned(strategy_returns, benchmark_returns)
    if joined.empty:
        return 0.0
    strat, bench = joined["s"], joined["b"]
    down = bench < 0
    if not bool(down.any()):
        return 0.0
    bench_down = float(bench[down].mean())
    if bench_down == 0:
        return 0.0
    return float(strat[down].mean()) / bench_down


def capture_ratio(strategy_returns: pd.Series, benchmark_returns: pd.Series) -> float:
    """Up-capture divided by down-capture.

    Greater than 1 is favourable (more upside than downside capture). Returns
    +inf when down-capture is zero with positive up-capture.
    """
    up = up_capture(strategy_returns, benchmark_returns)
    down = down_capture(strategy_returns, benchmark_returns)
    if down == 0:
        return float("inf") if up > 0 else 0.0
    return up / down


def rolling_alpha_beta(
    strategy_returns: pd.Series,
    benchmark_returns: pd.Series,
    window: int = 63,
    periods_per_year: int = 252,
) -> pd.DataFrame:
    """Rolling single-factor alpha and beta vs a benchmark.

    :func:`compute_beta` / :func:`factor_regression` give one full-sample
    number; this is the tearsheet's time-varying view — per bar, the OLS
    fit over the trailing ``window``::

        beta_t  = Cov(r_s, r_b) / Var(r_b)          (trailing window, ddof=1)
        alpha_t = (mean(r_s) - beta_t * mean(r_b)) * periods_per_year

    Args:
        strategy_returns: Per-bar strategy returns.
        benchmark_returns: Per-bar benchmark returns on the same index.
        window: Trailing OLS window (>= 2).
        periods_per_year: Bars per year for annualising the intercept.

    Returns:
        DataFrame with ``alpha`` (annualised) and ``beta`` columns; the
        warm-up and any window with zero benchmark variance are NaN.

    Raises:
        ValueError: If the indexes differ, ``window`` < 2 or
            ``periods_per_year`` < 1.
    """
    if not strategy_returns.index.equals(benchmark_returns.index):
        raise ValueError("strategy and benchmark returns must share the same index.")
    if window < 2:
        raise ValueError(f"window must be >= 2, got {window}.")
    if periods_per_year < 1:
        raise ValueError(f"periods_per_year must be >= 1, got {periods_per_year}.")

    cov = strategy_returns.rolling(window).cov(benchmark_returns)
    var = benchmark_returns.rolling(window).var(ddof=1)
    beta = (cov / var).replace([np.inf, -np.inf], np.nan)
    alpha = (
        strategy_returns.rolling(window).mean() - beta * benchmark_returns.rolling(window).mean()
    ) * periods_per_year
    return pd.DataFrame({"alpha": alpha, "beta": beta})
