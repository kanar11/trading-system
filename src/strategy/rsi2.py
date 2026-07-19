"""Connors RSI-2 pullback strategy.

Larry Connors' short-term mean-reversion system, one of the most widely
documented trading rules: on an instrument in a long-term up-trend, buy
sharp short-term pullbacks and sell them back into strength. The trend
filter (price vs its long moving average) is what separates it from a
naive "buy oversold" rule — pullbacks are only bought *with* the primary
trend, never against it.

Signal logic (package convention — decided on the close, shifted by the
backtest engine):

    up-trend  (close > SMA_trend):  long  when RSI(2) < entry_threshold,
                                    exit when close > SMA_exit
    down-trend (close < SMA_trend): short when RSI(2) > 100 - entry_threshold,
                                    exit when close < SMA_exit (if allow_short)

The 2-period RSI is extremely responsive, so ``entry_threshold`` is small
(classically 5-10). Reuses the shared ``rsi`` and ``sma`` primitives from
:mod:`src.indicators`.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src.indicators import rsi, sma

logger = logging.getLogger(__name__)


def rsi2_strategy(
    df: pd.DataFrame,
    rsi_period: int = 2,
    trend_window: int = 200,
    exit_window: int = 5,
    entry_threshold: float = 10.0,
    allow_short: bool = True,
) -> pd.DataFrame:
    """Generate Connors RSI-2 pullback signals.

    Args:
        df: DataFrame with at least a ``close`` column.
        rsi_period: RSI lookback (Connors' classic value is 2).
        trend_window: Long moving-average window defining the primary trend.
        exit_window: Short moving-average window for the exit rule.
        entry_threshold: RSI level below which an up-trend pullback is
            bought; ``100 - entry_threshold`` is the mirror short trigger.
            Must be in (0, 50).
        allow_short: If False, down-trend setups map to flat instead of −1.

    Returns:
        Copy of ``df`` with ``rsi``, ``sma_trend``, ``sma_exit`` and
        ``signal`` columns.

    Raises:
        ValueError: If ``close`` is missing or ``entry_threshold`` is not
            in (0, 50).
    """
    if "close" not in df.columns:
        raise ValueError("DataFrame must contain a 'close' column.")
    if not 0.0 < entry_threshold < 50.0:
        raise ValueError(f"entry_threshold must be in (0, 50), got {entry_threshold}.")

    df = df.copy()
    close = df["close"]
    df["rsi"] = rsi(close, period=rsi_period)
    df["sma_trend"] = sma(close, trend_window)
    df["sma_exit"] = sma(close, exit_window)

    close_v = close.to_numpy(dtype=float)
    rsi_v = df["rsi"].to_numpy(dtype=float)
    trend_v = df["sma_trend"].to_numpy(dtype=float)
    exit_v = df["sma_exit"].to_numpy(dtype=float)
    short_trigger = 100.0 - entry_threshold

    signal = np.zeros(len(df), dtype=int)
    state = 0
    for i in range(len(df)):
        if np.isnan(trend_v[i]) or np.isnan(rsi_v[i]) or np.isnan(exit_v[i]):
            state = 0  # warm-up: flat
            signal[i] = state
            continue

        uptrend = close_v[i] > trend_v[i]
        # exit rule: close a position once price crosses back through SMA_exit
        if (state == 1 and close_v[i] > exit_v[i]) or (state == -1 and close_v[i] < exit_v[i]):
            state = 0
        # look for a fresh entry only when flat
        if state == 0:
            if uptrend and rsi_v[i] < entry_threshold:
                state = 1
            elif (not uptrend) and allow_short and rsi_v[i] > short_trigger:
                state = -1
        signal[i] = state

    df["signal"] = signal
    return df
