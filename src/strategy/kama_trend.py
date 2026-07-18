"""KAMA adaptive trend-following strategy.

Classic moving-average trend rules whipsaw in chop because their lag is
fixed. Kaufman's own suggested use of his adaptive average
(:func:`src.indicators.kama`) is to trade the *price versus KAMA*
relationship: in a clean trend KAMA hugs price (fast smoothing constant)
and the signal follows promptly; in noise KAMA flattens (slow constant)
and small oscillations around it can be ignored with a dead band.

Signal logic per bar (package convention — decided on the close, shifted
by the backtest engine): from flat, enter on the plain close/KAMA
crossing; once positioned, the ``band`` acts as *hysteresis* — a long is
held until close falls below ``KAMA · (1 − band)`` (and symmetrically
for shorts), so small oscillations around the average do not flip the
position. ``band=0`` degenerates to the raw crossover.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src.indicators import kama

logger = logging.getLogger(__name__)


def kama_trend_strategy(
    df: pd.DataFrame,
    er_period: int = 10,
    fast: int = 2,
    slow: int = 30,
    band: float = 0.0,
    allow_short: bool = True,
) -> pd.DataFrame:
    """Generate adaptive trend signals from price versus KAMA.

    Args:
        df: DataFrame with a ``close`` column.
        er_period: Efficiency-Ratio window of the KAMA.
        fast: Fast EMA span bound of the adaptive constant.
        slow: Slow EMA span bound of the adaptive constant.
        band: Exit-hysteresis fraction (0.01 = 1%): an open position only
            flips once price crosses the opposite band edge, suppressing
            whipsaw around the average.
        allow_short: If False, downside breaks map to flat instead of −1.

    Returns:
        Copy of ``df`` with ``kama`` and ``signal`` columns.

    Raises:
        ValueError: If ``close`` is missing or ``band`` < 0 (KAMA parameter
            problems raise from the indicator).
    """
    if "close" not in df.columns:
        raise ValueError("DataFrame must contain a 'close' column.")
    if band < 0:
        raise ValueError(f"band must be >= 0, got {band}.")

    df = df.copy()
    adaptive = kama(df["close"], er_period=er_period, fast=fast, slow=slow)
    df["kama"] = adaptive

    close = df["close"].to_numpy(dtype=float)
    center = adaptive.to_numpy(dtype=float)
    upper = center * (1.0 + band)
    lower = center * (1.0 - band)

    signal = np.zeros(len(df), dtype=int)
    state = 0
    for i in range(len(df)):
        if np.isnan(center[i]):
            state = 0  # warm-up: flat
        elif state == 1:
            if close[i] < lower[i]:  # hysteresis exit for longs
                state = -1 if allow_short else 0
        elif state == -1:
            if close[i] > upper[i]:  # hysteresis exit for shorts
                state = 1
        elif close[i] > center[i]:  # from flat: plain crossover entries
            state = 1
        elif close[i] < center[i]:
            state = -1 if allow_short else 0
        signal[i] = state

    df["signal"] = signal
    return df
