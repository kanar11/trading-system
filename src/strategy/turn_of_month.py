"""Turn-of-month seasonal strategy.

One of the oldest documented equity anomalies (Lakonishok & Smidt 1988;
still measurable in index data): a disproportionate share of the market's
return accrues in the few trading days around the month boundary —
usually attributed to pension/payroll flows and window dressing. The
classic implementation is long the index only during that window (the
last trading day through the first three of the next month) and flat in
cash otherwise.

The window is located with :mod:`src.data.calendar`, so holidays and
missing sessions are handled from the bar index itself; the diagnostic
counterpart is :func:`src.reporting.seasonality.turn_of_month_effect`.
Long-only by construction (the anomaly is about *when* to hold, not
shorting the rest of the month); emits the package-standard ``signal``
column, decided on the bar's close and shifted by the backtest engine.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src.data.calendar import trading_day_of_month, trading_days_left_in_month

logger = logging.getLogger(__name__)


def turn_of_month_strategy(
    df: pd.DataFrame,
    days_before: int = 1,
    days_after: int = 3,
) -> pd.DataFrame:
    """Long-only signals for the turn-of-month window.

    Signal logic per bar:

        +1 (long) on the last ``days_before`` trading days of a month and
           on the first ``days_after`` trading days of a month
         0 (flat) otherwise

    The classic Lakonishok-Smidt window is ``days_before=1, days_after=3``
    (-1 to +3 around the boundary).

    Args:
        df: DataFrame with a ``close`` column on a sorted, unique
            DatetimeIndex.
        days_before: Trading days held *before* month-end (>= 0).
        days_after: Trading days held *after* month-start (>= 0).

    Returns:
        Copy of ``df`` with a ``signal`` column in {0, 1}.

    Raises:
        ValueError: If ``close`` is missing, a window is negative, or both
            windows are zero.
    """
    if "close" not in df.columns:
        raise ValueError("DataFrame must contain a 'close' column.")
    if days_before < 0 or days_after < 0:
        raise ValueError(f"days_before/days_after must be >= 0, got {days_before}/{days_after}.")
    if days_before == 0 and days_after == 0:
        raise ValueError("at least one of days_before/days_after must be > 0.")

    df = df.copy()
    index = pd.DatetimeIndex(df.index)
    day_no = trading_day_of_month(index).to_numpy()
    days_left = trading_days_left_in_month(index).to_numpy()

    in_window = (days_left < days_before) | (day_no <= days_after)
    df["signal"] = np.where(in_window, 1, 0)
    return df
