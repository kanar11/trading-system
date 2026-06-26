"""Pairs trading via Engle-Granger cointegration.

Standard stat-arb construction:

    1. Fit a hedge ratio β by OLS regression of asset Y on asset X.
    2. Form the spread  s_t = y_t - β · x_t.
    3. Test the spread for stationarity (cointegration). Trading is
       only justified if the spread is mean-reverting.
    4. Z-score the spread on a rolling window and trade extremes:
       short the spread when z > entry, long when z < -entry,
       exit when |z| < exit.

This module ships a *single-instrument* signal series indexed by date,
suitable for plugging into the existing backtest engine. The engine's
``signal`` is interpreted as "long the spread" (= long Y, short β·X)
when +1, "short the spread" when -1. For execution simulation that
respects both legs, run two backtests with the long/short halves and
combine, or use the synthetic spread series as the engine's ``close``.

The cointegration test is the standard Engle-Granger procedure: fit
the cointegration regression, then run an Augmented Dickey-Fuller test
on the residuals. We implement ADF from scratch (no statsmodels) using
the asymptotic 5% critical value from MacKinnon (2010) for the
residual-based test — appropriate for research-grade screening but
*not* a substitute for a full statistical package.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# MacKinnon (2010) 5% critical value for ADF on cointegration residuals
# with a constant and one regressor. Looser than the standard ADF 5%
# critical value precisely because we're testing a *fitted* residual.
ENGLE_GRANGER_5PCT = -3.34


@dataclass
class CointegrationResult:
    """Output of the Engle-Granger cointegration test.

    Attributes:
        hedge_ratio: OLS slope β from y = α + β · x + ε.
        intercept: OLS intercept α.
        adf_stat: ADF test statistic on the residuals.
        critical_5pct: 5% critical value for the residual-based test.
        is_cointegrated: True if ``adf_stat <= critical_5pct``.
        residuals: The estimated spread (y - α - β x).
    """

    hedge_ratio: float
    intercept: float
    adf_stat: float
    critical_5pct: float
    is_cointegrated: bool
    residuals: pd.Series


def _ols_simple(y: np.ndarray, x: np.ndarray) -> tuple[float, float]:
    """OLS slope + intercept of y on x. Returns (β, α)."""
    n = len(y)
    x_mean = x.mean()
    y_mean = y.mean()
    cov = ((x - x_mean) * (y - y_mean)).sum() / n
    var = ((x - x_mean) ** 2).sum() / n
    if var == 0:
        return 0.0, float(y_mean)
    beta = cov / var
    alpha = y_mean - beta * x_mean
    return float(beta), float(alpha)


def _adf_t_stat(series: np.ndarray, max_lag: int = 1) -> float:
    """Augmented Dickey-Fuller t-statistic for a unit root.

    Tests H0: φ = 1 (random walk) against H1: φ < 1 (stationary) by
    regressing Δs_t = γ s_{t-1} + Σ δ_i Δs_{t-i} + ε_t and returning
    the t-stat on γ. A *more negative* value rejects the unit-root
    null harder.

    Args:
        series: Input series (e.g. cointegration residuals).
        max_lag: Number of lagged differences to include (default 1).

    Returns:
        ADF t-statistic. Returns nan if the series is too short.
    """
    s = np.asarray(series, dtype=float)
    n = len(s)
    if n < max_lag + 5:
        return float("nan")

    ds = np.diff(s)
    n_eff = len(ds) - max_lag
    if n_eff < 5:
        return float("nan")

    y = ds[max_lag:]
    cols = [s[max_lag:-1]]  # s_{t-1}
    for lag in range(1, max_lag + 1):
        cols.append(ds[max_lag - lag : -lag])
    X = np.column_stack(cols)

    # OLS via lstsq
    coefs, *_ = np.linalg.lstsq(X, y, rcond=None)
    residuals = y - X @ coefs
    dof = max(len(y) - X.shape[1], 1)
    sigma2 = float((residuals**2).sum() / dof)
    try:
        xtx_inv = np.linalg.inv(X.T @ X)
    except np.linalg.LinAlgError:
        return float("nan")
    se_gamma = float(np.sqrt(sigma2 * xtx_inv[0, 0]))
    if se_gamma == 0:
        return float("nan")
    return float(coefs[0] / se_gamma)


def engle_granger_test(
    y: pd.Series,
    x: pd.Series,
    max_lag: int = 1,
    critical_value: float = ENGLE_GRANGER_5PCT,
) -> CointegrationResult:
    """Run the Engle-Granger cointegration test on two price series.

    Args:
        y: Dependent series.
        x: Independent series (regressor).
        max_lag: Number of lagged differences for the ADF stage.
        critical_value: Critical value to compare against (default
            5% MacKinnon for one regressor + constant).

    Returns:
        :class:`CointegrationResult`.
    """
    joined = pd.concat([y, x], axis=1, join="inner").dropna()
    if joined.shape[0] < max_lag + 10:
        raise ValueError(f"Need at least {max_lag + 10} aligned observations.")

    y_arr = joined.iloc[:, 0].values
    x_arr = joined.iloc[:, 1].values

    beta, alpha = _ols_simple(y_arr, x_arr)
    residuals = y_arr - alpha - beta * x_arr
    adf = _adf_t_stat(residuals, max_lag=max_lag)

    return CointegrationResult(
        hedge_ratio=beta,
        intercept=alpha,
        adf_stat=adf,
        critical_5pct=critical_value,
        is_cointegrated=bool(adf <= critical_value),
        residuals=pd.Series(residuals, index=joined.index, name="spread"),
    )


def pairs_trading_signal(
    y: pd.Series,
    x: pd.Series,
    z_window: int = 60,
    z_entry: float = 2.0,
    z_exit: float = 0.5,
    require_cointegration: bool = True,
) -> pd.DataFrame:
    """Build a pairs-trading signal series from two price streams.

    The strategy enters at ``|z| >= z_entry`` and exits when ``|z|
    falls below z_exit``. Positions are interpreted as "long the
    spread" (long Y, short β·X) when ``signal = +1`` and "short the
    spread" when ``signal = -1``. The output frame is single-asset
    and ready for the standard backtest engine, with the spread
    itself published as the ``close`` column.

    Args:
        y: First leg price series.
        x: Second leg price series.
        z_window: Rolling window for the z-score of the spread.
        z_entry: Absolute z-score that triggers entry.
        z_exit: Absolute z-score that triggers exit (closer to 0).
        require_cointegration: If True, raise unless the Engle-Granger
            test rejects the unit-root null at 5%.

    Returns:
        DataFrame with columns ``close`` (= spread), ``spread_z``,
        ``signal`` and a ``hedge_ratio`` metadata column.

    Raises:
        ValueError: If the series fail the cointegration test and
            ``require_cointegration`` is True.
    """
    test = engle_granger_test(y, x)
    if require_cointegration and not test.is_cointegrated:
        raise ValueError(
            f"Series are not cointegrated (ADF={test.adf_stat:.3f}, "
            f"5% critical={test.critical_5pct:.3f}). Set "
            f"require_cointegration=False to trade anyway."
        )

    spread = test.residuals
    rolling_mean = spread.rolling(z_window).mean()
    rolling_std = spread.rolling(z_window).std().replace(0, np.nan)
    z = (spread - rolling_mean) / rolling_std

    # generate state-machine signal
    signals = np.zeros(len(spread), dtype=int)
    state = 0
    z_arr = z.values
    for i in range(len(spread)):
        zi = z_arr[i]
        if np.isnan(zi):
            signals[i] = state
            continue
        if state == 0:
            if zi >= z_entry:
                state = -1  # spread too high → short the spread
            elif zi <= -z_entry:
                state = 1  # spread too low → long the spread
        else:
            if abs(zi) <= z_exit:
                state = 0
        signals[i] = state

    out = pd.DataFrame(
        {
            "close": spread.values,
            "spread_z": z.values,
            "signal": signals,
        },
        index=spread.index,
    )
    out.attrs["hedge_ratio"] = test.hedge_ratio
    out.attrs["intercept"] = test.intercept
    out.attrs["adf_stat"] = test.adf_stat
    return out
