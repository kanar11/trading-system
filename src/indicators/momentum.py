"""Momentum / oscillator indicators."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.indicators.trend import ema


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Relative Strength Index (Wilder's smoothing)."""
    if period <= 1:
        raise ValueError("period must be >= 2")
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def macd(
    close: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> pd.DataFrame:
    """MACD line, signal line, and histogram.

    Returns a DataFrame with columns: ``macd``, ``signal``, ``hist``.
    """
    if fast >= slow:
        raise ValueError(f"fast ({fast}) must be < slow ({slow})")
    macd_line = ema(close, fast) - ema(close, slow)
    sig_line = ema(macd_line, signal)
    return pd.DataFrame({"macd": macd_line, "signal": sig_line, "hist": macd_line - sig_line})


def stochastic(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    k_period: int = 14,
    d_period: int = 3,
) -> pd.DataFrame:
    """Stochastic %K and %D.

    Returns a DataFrame with columns: ``k``, ``d``.
    """
    lowest = low.rolling(k_period, min_periods=k_period).min()
    highest = high.rolling(k_period, min_periods=k_period).max()
    denom = (highest - lowest).replace(0, np.nan)
    k = 100 * (close - lowest) / denom
    d = k.rolling(d_period, min_periods=d_period).mean()
    return pd.DataFrame({"k": k, "d": d})


def williams_r(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14,
) -> pd.Series:
    """Williams %R — equivalent to (100 - %K) on a negative scale.

    Returns values in [-100, 0].
    """
    highest = high.rolling(period, min_periods=period).max()
    lowest = low.rolling(period, min_periods=period).min()
    denom = (highest - lowest).replace(0, np.nan)
    return -100 * (highest - close) / denom


def cci(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 20,
    constant: float = 0.015,
) -> pd.Series:
    """Commodity Channel Index.

    CCI = (TP - SMA(TP)) / (constant * mean_dev(TP)), where TP = typical price.
    """
    tp = (high + low + close) / 3
    sma_tp = tp.rolling(period, min_periods=period).mean()
    mean_dev = (tp - sma_tp).abs().rolling(period, min_periods=period).mean()
    denom = constant * mean_dev.replace(0, np.nan)
    return (tp - sma_tp) / denom


def roc(close: pd.Series, period: int = 10) -> pd.Series:
    """Rate of change (percentage)."""
    if period <= 0:
        raise ValueError("period must be > 0")
    return 100 * close.pct_change(period)


def trix(close: pd.Series, period: int = 15) -> pd.Series:
    """TRIX — rate of change of a triple-smoothed EMA (percent).

    Oscillates around zero: positive in up-trends, negative in down-trends.
    """
    if period <= 0:
        raise ValueError("period must be > 0")
    triple = ema(ema(ema(close, period), period), period)
    return 100 * triple.pct_change()


def cmo(close: pd.Series, period: int = 14) -> pd.Series:
    """Chande Momentum Oscillator, in [-100, 100].

    CMO = 100 * (sum_up - sum_down) / (sum_up + sum_down) over ``period`` bars.
    """
    if period <= 0:
        raise ValueError("period must be > 0")
    delta = close.diff()
    up = delta.clip(lower=0.0)
    down = (-delta).clip(lower=0.0)
    sum_up = up.rolling(period, min_periods=period).sum()
    sum_down = down.rolling(period, min_periods=period).sum()
    denom = (sum_up + sum_down).replace(0, np.nan)
    return 100 * (sum_up - sum_down) / denom


def elder_ray(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 13,
) -> pd.DataFrame:
    """Elder-Ray bull / bear power around an EMA baseline.

        bull_power = high - EMA(close, period)   (buyers' strength, > 0 in up-trends)
        bear_power = low  - EMA(close, period)   (sellers' strength, < 0 in down-trends)

    Returns a DataFrame with ``bull_power`` and ``bear_power`` columns.

    Raises:
        ValueError: If ``period`` < 1.
    """
    if period < 1:
        raise ValueError(f"period must be >= 1, got {period}.")
    baseline = ema(close, period)
    return pd.DataFrame({"bull_power": high - baseline, "bear_power": low - baseline})


def lookback_return(close: pd.Series, lookback: int = 252, skip: int = 21) -> pd.Series:
    """Skip-month momentum score (the industry "12-1" convention).

    The cross-sectional and time-series momentum literature (Jegadeesh &
    Titman 1993; Moskowitz, Ooi & Pedersen 2012) measures momentum as the
    return over the past ``lookback`` bars *excluding* the most recent
    ``skip`` bars, side-stepping the well-documented 1-month short-term
    reversal::

        score_t = close[t - skip] / close[t - lookback] - 1

    With the defaults (252, 21) this is the classic 12-1 month score on
    daily bars; ``skip=0`` gives the plain trailing return (like
    :func:`roc` but expressed as a fraction, not a percentage).

    Raises:
        ValueError: If ``skip`` < 0 or ``lookback`` <= ``skip``.
    """
    if skip < 0:
        raise ValueError(f"skip must be >= 0, got {skip}.")
    if lookback <= skip:
        raise ValueError(f"lookback ({lookback}) must be > skip ({skip}).")
    return close.shift(skip) / close.shift(lookback) - 1


def distance_from_high(close: pd.Series, window: int = 252) -> pd.Series:
    """Fractional distance below the rolling ``window`` high (<= 0).

    George & Hwang (2004) show proximity to the 52-week high is itself a
    momentum signal: ``close / rolling_max(close) - 1`` is 0 at a fresh
    high and increasingly negative the further price sits below it.

    Raises:
        ValueError: If ``window`` < 1.
    """
    if window < 1:
        raise ValueError(f"window must be >= 1, got {window}.")
    return close / close.rolling(window, min_periods=window).max() - 1
