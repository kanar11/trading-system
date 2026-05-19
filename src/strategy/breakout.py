"""Donchian-channel breakout strategy.

Generates trading signals on N-day high / low breakouts, with an
optional ATR-based volatility filter to suppress signals when the
move is small relative to recent range.

The classic "Turtle"-style entry: go long on a new N-day high, short
on a new N-day low, exit when price crosses an opposite shorter
channel (M-day low/high). This is a pure-trend follower and works
best in regimes with sustained directional moves.
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _atr(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14,
) -> pd.Series:
    """Average True Range over ``period`` bars."""
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return true_range.rolling(period).mean()


def breakout_strategy(
    df: pd.DataFrame,
    entry_window: int = 20,
    exit_window: int = 10,
    atr_period: int = 14,
    atr_filter: float = 0.0,
    allow_short: bool = True,
) -> pd.DataFrame:
    """Generate Donchian-channel breakout signals.

    Signal logic (evaluated on the close of the bar):
        +1 (long)   when close >= rolling N-day high (entry_window)
        -1 (short)  when close <= rolling N-day low (entry_window)
        long exit   when close <= rolling M-day low  (exit_window)
        short exit  when close >= rolling M-day high (exit_window)
         0 otherwise

    The ATR filter is optional: signals are suppressed when the
    breakout size (close minus prior channel level) is less than
    ``atr_filter`` * ATR. Set ``atr_filter=0`` to disable.

    Args:
        df: DataFrame with at least 'close'. 'high' / 'low' are used
            when present, otherwise close is used as a proxy.
        entry_window: Lookback for the entry channel (long N-day high,
            short N-day low). Excludes the current bar.
        exit_window: Lookback for the opposite exit channel. Should be
            smaller than ``entry_window`` for classic turtle behaviour.
        atr_period: Period for the ATR filter.
        atr_filter: Minimum breakout size in ATR units (0 disables).
        allow_short: If False, short signals are clamped to flat.

    Returns:
        DataFrame with added indicator columns and a 'signal' column.
    """
    df = df.copy()

    high = df["high"] if "high" in df.columns else df["close"]
    low = df["low"] if "low" in df.columns else df["close"]
    close = df["close"]

    # exclude the current bar from the rolling channel (shift by 1)
    df["entry_high"] = high.shift(1).rolling(entry_window).max()
    df["entry_low"] = low.shift(1).rolling(entry_window).min()
    df["exit_high"] = high.shift(1).rolling(exit_window).max()
    df["exit_low"] = low.shift(1).rolling(exit_window).min()
    df["atr"] = _atr(high, low, close, period=atr_period)

    long_break = close >= df["entry_high"]
    short_break = close <= df["entry_low"]

    if atr_filter > 0:
        size_up = (close - df["entry_high"]).abs()
        size_dn = (df["entry_low"] - close).abs()
        long_break &= size_up >= atr_filter * df["atr"]
        short_break &= size_dn >= atr_filter * df["atr"]

    # build position by walking forward — Donchian needs state for exits
    n = len(df)
    position = np.zeros(n, dtype=int)
    state = 0
    closes = close.values
    exit_low = df["exit_low"].values
    exit_high = df["exit_high"].values
    long_arr = long_break.fillna(False).values
    short_arr = short_break.fillna(False).values

    for i in range(n):
        # exit first — Donchian exit channel breach
        if state == 1 and not np.isnan(exit_low[i]) and closes[i] <= exit_low[i]:
            state = 0
        elif state == -1 and not np.isnan(exit_high[i]) and closes[i] >= exit_high[i]:
            state = 0

        # then check entry
        if state == 0:
            if long_arr[i]:
                state = 1
            elif short_arr[i] and allow_short:
                state = -1

        position[i] = state

    df["signal"] = position
    return df
