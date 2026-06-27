"""Trend-following moving averages."""

from __future__ import annotations

import numpy as np
import pandas as pd


def sma(series: pd.Series, window: int) -> pd.Series:
    """Simple moving average."""
    if window <= 0:
        raise ValueError("window must be > 0")
    return series.rolling(window, min_periods=window).mean()


def ema(series: pd.Series, span: int) -> pd.Series:
    """Exponential moving average (pandas span convention)."""
    if span <= 0:
        raise ValueError("span must be > 0")
    return series.ewm(span=span, adjust=False, min_periods=span).mean()


def wma(series: pd.Series, window: int) -> pd.Series:
    """Linearly-weighted moving average (most recent bar weighted most heavily)."""
    if window <= 0:
        raise ValueError("window must be > 0")
    weights = np.arange(1, window + 1, dtype=float)
    weight_sum = weights.sum()

    def _wma_window(x: np.ndarray) -> float:
        return float(np.dot(x, weights) / weight_sum)

    return series.rolling(window, min_periods=window).apply(_wma_window, raw=True)


def vwma(price: pd.Series, volume: pd.Series, window: int) -> pd.Series:
    """Volume-weighted moving average over a rolling window."""
    if window <= 0:
        raise ValueError("window must be > 0")
    pv = price * volume
    return (
        pv.rolling(window, min_periods=window).sum()
        / volume.rolling(window, min_periods=window).sum()
    )


def hma(series: pd.Series, window: int) -> pd.Series:
    """Hull Moving Average — a low-lag, smooth trend filter.

    HMA(n) = WMA( 2 * WMA(n / 2) - WMA(n), round(sqrt(n)) ).
    """
    if window <= 1:
        raise ValueError("window must be >= 2")
    half = max(window // 2, 1)
    sqrt_window = max(int(round(window**0.5)), 1)
    raw = 2 * wma(series, half) - wma(series, window)
    return wma(raw, sqrt_window)


def aroon(high: pd.Series, low: pd.Series, period: int = 25) -> pd.DataFrame:
    """Aroon up / down / oscillator.

    Aroon Up measures how recently the rolling ``period``-bar high occurred
    (100 = the high is the current bar, 0 = it was ``period`` bars ago); Aroon
    Down does the same for the low. The oscillator is up - down, in [-100, 100].

    Returns a DataFrame with columns: ``up``, ``down``, ``oscillator``.
    """
    if period <= 0:
        raise ValueError("period must be > 0")

    def _bars_since_high(x: np.ndarray) -> float:
        return float(len(x) - 1 - np.argmax(x))

    def _bars_since_low(x: np.ndarray) -> float:
        return float(len(x) - 1 - np.argmin(x))

    win = period + 1
    high_age = high.rolling(win, min_periods=win).apply(_bars_since_high, raw=True)
    low_age = low.rolling(win, min_periods=win).apply(_bars_since_low, raw=True)
    up = 100 * (period - high_age) / period
    down = 100 * (period - low_age) / period
    return pd.DataFrame({"up": up, "down": down, "oscillator": up - down})
