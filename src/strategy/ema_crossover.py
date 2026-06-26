"""Exponential-moving-average and MACD crossover strategies.

Two related signals that pre-date this codebase by several decades but
remain a standard reference baseline for trend-following research:

    * ``ema_crossover_strategy``  — fast EMA vs slow EMA. Long when fast
      is above slow, short when below, optional flat-zone gap.

    * ``macd_strategy``           — classical MACD (12, 26, 9). Long
      when the MACD line crosses above its signal line, short on a
      cross below. A histogram column is exposed for plotting.

Both share the same EMA primitive and expose enough intermediate
columns for downstream diagnostics.
"""

from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)


def _ema(series: pd.Series, span: int) -> pd.Series:
    """Standard exponential moving average."""
    return series.ewm(span=span, adjust=False, min_periods=span).mean()


def ema_crossover_strategy(
    df: pd.DataFrame,
    fast: int = 20,
    slow: int = 50,
    gap_bps: float = 0.0,
    allow_short: bool = True,
) -> pd.DataFrame:
    """Fast-vs-slow EMA crossover signals.

    Signal logic:
        +1 (long)  when fast EMA > slow EMA * (1 + gap)
        -1 (short) when fast EMA < slow EMA * (1 - gap)
         0 (flat)  inside the gap (avoids signal churn around equality)

    Args:
        df: DataFrame with a 'close' column.
        fast: Fast EMA span (must be < slow).
        slow: Slow EMA span.
        gap_bps: Minimum spread between EMAs (in basis points) required
            to take a position. Zero disables the gap.
        allow_short: If False, short signals become flat.

    Returns:
        DataFrame with ``ema_fast``, ``ema_slow`` and ``signal`` columns.
    """
    if fast >= slow:
        raise ValueError(f"fast ({fast}) must be < slow ({slow})")

    df = df.copy()
    df["ema_fast"] = _ema(df["close"], fast)
    df["ema_slow"] = _ema(df["close"], slow)

    gap = gap_bps / 10_000.0
    long_cond = df["ema_fast"] > df["ema_slow"] * (1 + gap)
    short_cond = df["ema_fast"] < df["ema_slow"] * (1 - gap)

    df["signal"] = 0
    df.loc[long_cond, "signal"] = 1
    if allow_short:
        df.loc[short_cond, "signal"] = -1

    return df


def macd_strategy(
    df: pd.DataFrame,
    fast: int = 12,
    slow: int = 26,
    signal_span: int = 9,
    allow_short: bool = True,
) -> pd.DataFrame:
    """Classical MACD crossover signals.

    The MACD line is ``EMA_fast(close) − EMA_slow(close)``. The
    signal line is ``EMA(MACD, signal_span)``. The strategy goes long
    when MACD crosses above the signal line and short when it crosses
    below. The histogram (MACD − signal) is exposed for plotting.

    Args:
        df: DataFrame with a 'close' column.
        fast: Fast EMA span for MACD line.
        slow: Slow EMA span for MACD line.
        signal_span: EMA span of the MACD signal line.
        allow_short: If False, short signals are clamped to flat.

    Returns:
        DataFrame with ``macd``, ``macd_signal``, ``macd_hist`` and
        ``signal`` columns.
    """
    if fast >= slow:
        raise ValueError(f"fast ({fast}) must be < slow ({slow})")

    df = df.copy()
    macd = _ema(df["close"], fast) - _ema(df["close"], slow)
    signal_line = _ema(macd, signal_span)
    hist = macd - signal_line

    df["macd"] = macd
    df["macd_signal"] = signal_line
    df["macd_hist"] = hist

    long_cond = hist > 0
    short_cond = hist < 0

    df["signal"] = 0
    df.loc[long_cond, "signal"] = 1
    if allow_short:
        df.loc[short_cond, "signal"] = -1

    return df
