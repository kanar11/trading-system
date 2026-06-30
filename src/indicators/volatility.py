"""Volatility / range-based indicators."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.indicators.trend import ema, sma


def _true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    return pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)


def atr(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14,
    smoothing: str = "ema",
) -> pd.Series:
    """Average True Range.

    Args:
        high, low, close: Price series.
        period: Lookback period.
        smoothing: ``"ema"`` (Wilder via EMA, default), ``"sma"``, or
            ``"wilder"`` for the original Wilder RMA.
    """
    tr = _true_range(high, low, close)
    if smoothing == "sma":
        return tr.rolling(period, min_periods=period).mean()
    if smoothing == "ema":
        return ema(tr, period)
    if smoothing == "wilder":
        return tr.ewm(alpha=1 / period, min_periods=period).mean()
    raise ValueError(f"unknown smoothing: {smoothing!r}")


def bollinger(
    close: pd.Series,
    window: int = 20,
    num_std: float = 2.0,
) -> pd.DataFrame:
    """Bollinger Bands.

    Returns a DataFrame with columns ``middle``, ``upper``, ``lower``,
    ``bandwidth`` (= (upper - lower) / middle) and ``percent_b``
    (= (close - lower) / (upper - lower)).
    """
    middle = sma(close, window)
    std = close.rolling(window, min_periods=window).std()
    upper = middle + num_std * std
    lower = middle - num_std * std
    bandwidth = (upper - lower) / middle
    pct_b = (close - lower) / (upper - lower).replace(0, pd.NA)
    return pd.DataFrame(
        {
            "middle": middle,
            "upper": upper,
            "lower": lower,
            "bandwidth": bandwidth,
            "percent_b": pct_b,
        }
    )


def keltner(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    window: int = 20,
    atr_period: int = 10,
    atr_mult: float = 2.0,
) -> pd.DataFrame:
    """Keltner Channels: EMA + ATR-based bands.

    Returns ``middle``, ``upper``, ``lower``.
    """
    middle = ema(close, window)
    atr_v = atr(high, low, close, period=atr_period)
    return pd.DataFrame(
        {
            "middle": middle,
            "upper": middle + atr_mult * atr_v,
            "lower": middle - atr_mult * atr_v,
        }
    )


def donchian(
    high: pd.Series,
    low: pd.Series,
    window: int = 20,
) -> pd.DataFrame:
    """Donchian Channels: rolling N-day high / low. Returns ``upper``, ``lower``, ``middle``."""
    upper = high.rolling(window, min_periods=window).max()
    lower = low.rolling(window, min_periods=window).min()
    return pd.DataFrame({"upper": upper, "lower": lower, "middle": (upper + lower) / 2})


def supertrend(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 10,
    multiplier: float = 3.0,
) -> pd.DataFrame:
    """SuperTrend: an ATR-banded trailing trend follower.

    Bands are ``(high + low) / 2 +/- multiplier * ATR``; the line latches to the
    lower band in an uptrend and the upper band in a downtrend, flipping when
    price closes through it.

    Returns a DataFrame with:
        * ``supertrend`` — the trailing line (NaN during the ATR warm-up).
        * ``direction``  — +1 in an uptrend (line below price), -1 in a downtrend.
    """
    if period < 1:
        raise ValueError(f"period must be >= 1, got {period}.")

    atr_v = atr(high, low, close, period=period)
    hl2 = (high + low) / 2
    upper = (hl2 + multiplier * atr_v).to_numpy()
    lower = (hl2 - multiplier * atr_v).to_numpy()
    close_arr = close.to_numpy()
    n = len(close)

    line = np.full(n, np.nan)
    direction = np.zeros(n, dtype=int)
    final_upper = np.full(n, np.nan)
    final_lower = np.full(n, np.nan)

    for i in range(n):
        if np.isnan(upper[i]) or np.isnan(lower[i]):
            continue
        if i == 0 or np.isnan(final_upper[i - 1]):
            final_upper[i] = upper[i]
            final_lower[i] = lower[i]
            up = close_arr[i] > final_lower[i]
            line[i] = final_lower[i] if up else final_upper[i]
            direction[i] = 1 if up else -1
            continue

        final_upper[i] = (
            upper[i]
            if (upper[i] < final_upper[i - 1] or close_arr[i - 1] > final_upper[i - 1])
            else final_upper[i - 1]
        )
        final_lower[i] = (
            lower[i]
            if (lower[i] > final_lower[i - 1] or close_arr[i - 1] < final_lower[i - 1])
            else final_lower[i - 1]
        )

        if line[i - 1] == final_upper[i - 1]:
            up = close_arr[i] > final_upper[i]
        else:
            up = close_arr[i] >= final_lower[i]
        line[i] = final_lower[i] if up else final_upper[i]
        direction[i] = 1 if up else -1

    return pd.DataFrame({"supertrend": line, "direction": direction}, index=close.index)
