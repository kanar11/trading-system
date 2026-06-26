"""Volatility / range-based indicators."""

from __future__ import annotations

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
