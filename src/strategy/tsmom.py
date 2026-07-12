"""Time-series momentum (TSMOM) strategy.

The single-asset momentum benchmark of the industry, after Moskowitz, Ooi
& Pedersen (2012): go long when the asset's own skip-month lookback return
(the "12-1" score) is positive, short when negative — no cross-sectional
ranking, just the asset against its own history. On an equity index like
SPY this is the classic trend-following overlay whose returns look like a
long-straddle payoff in market-timing regressions.

Decisions are made on periodic rebalance dates (month-end by default, via
:mod:`src.data.calendar`) and held until the next one; the momentum score
comes from :func:`src.indicators.lookback_return`. The ``signal`` column
follows the package convention — decided on the bar's close, shifted for
execution by the backtest engine.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src.data.calendar import rebalance_mask
from src.indicators import lookback_return

logger = logging.getLogger(__name__)


def tsmom_strategy(
    df: pd.DataFrame,
    lookback: int = 252,
    skip: int = 21,
    rebalance: str = "M",
    allow_short: bool = True,
) -> pd.DataFrame:
    """Generate time-series momentum signals for a single asset.

    Signal logic, evaluated on each rebalance date (last bar of the
    ``rebalance`` period) and held until the next one:

        +1 (long)  when the skip-month lookback return > 0
        -1 (short) when it is < 0 (0 instead when ``allow_short=False``)
         0 (flat)  while the score is still NaN (warm-up)

    Args:
        df: DataFrame with at least a ``close`` column on a sorted,
            unique DatetimeIndex.
        lookback: Momentum lookback in bars (252 = 12 months of days).
        skip: Most-recent bars excluded from the score (21 = skip-month).
        rebalance: Decision period — ``"W"``, ``"M"``, ``"Q"`` or ``"Y"``.
        allow_short: If False, negative momentum maps to flat, giving the
            long-only trend overlay common for equity indexes.

    Returns:
        Copy of ``df`` with ``momentum_score`` and ``signal`` columns.

    Raises:
        ValueError: If ``close`` is missing (score/calendar parameter
            problems raise from the underlying helpers).
    """
    if "close" not in df.columns:
        raise ValueError("DataFrame must contain a 'close' column.")

    df = df.copy()
    index = pd.DatetimeIndex(df.index)
    score = lookback_return(df["close"], lookback=lookback, skip=skip)
    df["momentum_score"] = score

    decision_bars = rebalance_mask(index, freq=rebalance, which="last").to_numpy()
    score_values = score.to_numpy(dtype=float)

    signal = np.full(len(df), np.nan)
    for pos in np.flatnonzero(decision_bars):
        value = score_values[pos]
        if np.isnan(value):
            signal[pos] = 0.0  # warm-up: stay flat
        elif value > 0:
            signal[pos] = 1.0
        else:
            signal[pos] = -1.0 if allow_short else 0.0

    # hold each decision until the next rebalance; flat before the first
    df["signal"] = pd.Series(signal, index=df.index).ffill().fillna(0.0).astype(int)
    return df
