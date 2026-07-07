"""Calendar-seasonality reports over a return series.

Answers the classic seasonality questions about a strategy or asset return
stream: which calendar months carry the performance (January effect, "sell
in May"), which weekdays (Monday/Friday effects), and whether returns
cluster around the month boundary (turn-of-month effect). Each report gives
the mean return, the hit rate (fraction of positive observations) and the
observation count, so thin sample sizes are visible at a glance.

Monthly statistics are computed on *compounded* month returns (so a 21-bar
month counts once, not 21 times); weekday and turn-of-month statistics are
per-bar. Turn-of-month bars are located with :mod:`src.data.calendar`, so
holidays and missing sessions are handled automatically. Direct-import
module::

    from src.reporting.seasonality import monthly_seasonality
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.calendar import trading_day_of_month, trading_days_left_in_month

_DAY_NAMES = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


def _validate_returns(returns: pd.Series) -> pd.DatetimeIndex:
    """Require a non-empty return series on a DatetimeIndex."""
    if not isinstance(returns.index, pd.DatetimeIndex):
        raise TypeError(f"returns must have a DatetimeIndex, got {type(returns.index).__name__}.")
    if len(returns) == 0:
        raise ValueError("returns must not be empty.")
    return returns.index


def _stats_by_group(values: pd.Series, keys: np.ndarray) -> pd.DataFrame:
    """Mean, hit rate and count of ``values`` grouped by ``keys``."""
    grouped = values.groupby(keys)
    return pd.DataFrame(
        {
            "mean_return": grouped.mean(),
            "hit_rate": (values > 0).groupby(keys).mean(),
            "n_obs": grouped.size(),
        }
    )


def monthly_seasonality(returns: pd.Series) -> pd.DataFrame:
    """Average compounded return per calendar month (Jan..Dec).

    Bar returns are compounded within each (year, month) first, so every
    month contributes one observation regardless of its bar count.

    Args:
        returns: Per-bar returns on a DatetimeIndex.

    Returns:
        DataFrame indexed by month number (1-12, only months present) with
        ``mean_return``, ``hit_rate`` and ``n_obs`` columns.

    Raises:
        TypeError: If the index is not a DatetimeIndex.
        ValueError: If ``returns`` is empty.
    """
    index = _validate_returns(returns)
    periods = index.to_period("M")
    compounded = (1.0 + returns).groupby(periods).prod() - 1.0
    month_numbers = np.asarray(pd.PeriodIndex(compounded.index).month)
    table = _stats_by_group(compounded, month_numbers)
    table.index.name = "month"
    return table


def day_of_week_seasonality(returns: pd.Series) -> pd.DataFrame:
    """Average per-bar return by weekday.

    Args:
        returns: Per-bar returns on a DatetimeIndex.

    Returns:
        DataFrame indexed by day name (Monday..Sunday, only days present)
        with ``mean_return``, ``hit_rate`` and ``n_obs`` columns.

    Raises:
        TypeError: If the index is not a DatetimeIndex.
        ValueError: If ``returns`` is empty.
    """
    index = _validate_returns(returns)
    table = _stats_by_group(returns, np.asarray(index.dayofweek))
    table.index = pd.Index([_DAY_NAMES[int(i)] for i in table.index], name="day")
    return table


def turn_of_month_effect(returns: pd.Series, window: int = 3) -> pd.DataFrame:
    """Per-bar return statistics at the month boundary vs the rest.

    A bar is "turn of month" when it is among the last ``window`` trading
    days of a month or the first ``window`` of the next (both boundaries of
    the same turn), counted on the bars actually present in the index.

    Args:
        returns: Per-bar returns on a sorted, unique DatetimeIndex.
        window: Boundary width in trading days (>= 1).

    Returns:
        DataFrame with rows ``"turn_of_month"`` and ``"other"`` and
        ``mean_return``, ``hit_rate``, ``n_obs`` columns.

    Raises:
        TypeError: If the index is not a DatetimeIndex.
        ValueError: If ``returns`` is empty or ``window`` < 1.
    """
    index = _validate_returns(returns)
    if window < 1:
        raise ValueError(f"window must be >= 1, got {window}.")

    day_no = trading_day_of_month(index).to_numpy()
    days_left = trading_days_left_in_month(index).to_numpy()
    is_turn = (day_no <= window) | (days_left < window)

    labels = np.where(is_turn, "turn_of_month", "other")
    table = _stats_by_group(returns, labels)
    table.index.name = "period"
    return table.reindex([label for label in ("turn_of_month", "other") if label in table.index])
