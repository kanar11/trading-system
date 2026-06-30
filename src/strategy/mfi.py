"""Money Flow Index mean-reversion strategy.

A volume-aware counterpart to the Bollinger/RSI mean-reversion strategy: the
Money Flow Index (a volume-weighted RSI) enters against extremes — long when
money flow is washed out (MFI < oversold), short when it is overheated
(MFI > overbought) — and holds until the index reverts to a neutral level.

Reuses the shared ``mfi`` primitive from :mod:`src.indicators`.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src.indicators import mfi

logger = logging.getLogger(__name__)

_REQUIRED = ("high", "low", "close", "volume")


def mfi_strategy(
    df: pd.DataFrame,
    period: int = 14,
    oversold: float = 20.0,
    overbought: float = 80.0,
    exit_level: float = 50.0,
    allow_short: bool = True,
) -> pd.DataFrame:
    """Money Flow Index mean-reversion signals.

    Signal logic (stateful, held until reversion):
        enter +1 (long)  when MFI < oversold; exit when MFI >= exit_level
        enter -1 (short) when MFI > overbought; exit when MFI <= exit_level
         0 (flat)        otherwise (including the warm-up window)

    Args:
        df: DataFrame with 'high', 'low', 'close' and 'volume' columns.
        period: MFI lookback period.
        oversold: MFI level below which to go long.
        overbought: MFI level above which to go short.
        exit_level: Neutral MFI level at which an open position is closed.
        allow_short: If False, short entries are suppressed (long/flat only).

    Returns:
        DataFrame with an added ``mfi`` column and a ``signal`` column.

    Raises:
        ValueError: If a required column is missing, ``period`` < 1, or the
            level thresholds are not ordered oversold < exit_level < overbought.
    """
    missing = [c for c in _REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"DataFrame must contain columns {missing}.")
    if period < 1:
        raise ValueError(f"period must be >= 1, got {period}.")
    if not oversold < exit_level < overbought:
        raise ValueError(
            f"levels must satisfy oversold ({oversold}) < exit_level "
            f"({exit_level}) < overbought ({overbought})."
        )

    df = df.copy()
    mfi_values = mfi(df["high"], df["low"], df["close"], df["volume"], period)
    df["mfi"] = mfi_values

    values = mfi_values.to_numpy()
    signals = np.zeros(len(df), dtype=int)
    position = 0
    for i in range(len(df)):
        v = values[i]
        if np.isnan(v):
            position = 0
        elif position == 0:
            if v < oversold:
                position = 1
            elif v > overbought and allow_short:
                position = -1
        elif position == 1 and v >= exit_level or position == -1 and v <= exit_level:
            position = 0
        signals[i] = position

    df["signal"] = signals
    return df
