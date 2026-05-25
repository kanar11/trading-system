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
