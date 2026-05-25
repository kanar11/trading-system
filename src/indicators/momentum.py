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
    return pd.DataFrame(
        {"macd": macd_line, "signal": sig_line, "hist": macd_line - sig_line}
    )


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
