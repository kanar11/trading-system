"""Trading-calendar helpers over a bar index.

Rebalancing and seasonality logic keeps re-deriving the same facts from a
``DatetimeIndex``: which bars are the first/last trading day of a week,
month, quarter or year, and how deep into the month a bar sits. These
helpers answer those questions *from the index itself* — the calendar is
whatever bars actually exist in the data, so exchange holidays and missing
sessions are handled for free, with no external calendar dependency.

All functions require a sorted, duplicate-free :class:`pandas.DatetimeIndex`
and are pure (no mutation). Direct-import module::

    from src.data.calendar import rebalance_mask, rebalance_dates
"""

from __future__ import annotations

import numpy as np
import pandas as pd

_FREQ_CODES = {"W": "W", "M": "M", "Q": "Q", "Y": "Y"}


def _validate_index(index: pd.DatetimeIndex) -> None:
    """Reject anything that is not a sorted, unique DatetimeIndex."""
    if not isinstance(index, pd.DatetimeIndex):
        raise TypeError(f"index must be a DatetimeIndex, got {type(index).__name__}.")
    if not index.is_monotonic_increasing:
        raise ValueError("index must be sorted in increasing order.")
    if index.has_duplicates:
        raise ValueError("index must not contain duplicate timestamps.")


def _period_boundaries(index: pd.DatetimeIndex, freq: str) -> np.ndarray:
    """Boolean array marking bars whose *next* bar starts a new period."""
    key = freq.strip().upper()
    if key not in _FREQ_CODES:
        raise ValueError(f"freq must be one of {sorted(_FREQ_CODES)}, got {freq!r}.")
    periods = index.to_period(_FREQ_CODES[key])
    return np.asarray(periods[1:] != periods[:-1])


def rebalance_mask(
    index: pd.DatetimeIndex,
    freq: str = "M",
    which: str = "last",
) -> pd.Series:
    """Boolean mask of the first/last trading day per calendar period.

    Args:
        index: Sorted, unique bar timestamps.
        freq: Calendar period — ``"W"``, ``"M"``, ``"Q"`` or ``"Y"``.
        which: ``"last"`` (default) marks the final bar of each period,
            ``"first"`` the opening bar.

    Returns:
        Boolean Series named ``"rebalance"`` aligned to ``index``.

    Raises:
        TypeError: If ``index`` is not a DatetimeIndex.
        ValueError: If the index is unsorted/duplicated or ``freq`` /
            ``which`` is unknown.
    """
    _validate_index(index)
    if which not in ("first", "last"):
        raise ValueError(f"which must be 'first' or 'last', got {which!r}.")
    if len(index) == 0:
        return pd.Series(np.zeros(0, dtype=bool), index=index, name="rebalance")

    boundary = _period_boundaries(index, freq)
    # the final bar always closes a period; the first bar always opens one
    mask = np.append(boundary, True) if which == "last" else np.insert(boundary, 0, True)
    return pd.Series(mask, index=index, name="rebalance")


def rebalance_dates(
    index: pd.DatetimeIndex,
    freq: str = "M",
    which: str = "last",
) -> pd.DatetimeIndex:
    """Timestamps of the first/last trading day per period (see
    :func:`rebalance_mask`)."""
    return index[rebalance_mask(index, freq=freq, which=which).to_numpy()]


def trading_day_of_month(index: pd.DatetimeIndex) -> pd.Series:
    """1-based trading-day counter within each calendar month.

    The first bar of a month is 1, the next 2, and so on — the standard
    input for turn-of-month seasonality rules.

    Returns:
        Integer Series named ``"trading_day"`` aligned to ``index``.
    """
    _validate_index(index)
    n = len(index)
    if n == 0:
        return pd.Series(np.zeros(0, dtype=int), index=index, name="trading_day")

    first = np.insert(_period_boundaries(index, "M"), 0, True)
    positions = np.arange(n)
    starts = positions[first]
    sizes = np.diff(np.append(starts, n))
    month_start = np.repeat(starts, sizes)
    return pd.Series(positions - month_start + 1, index=index, name="trading_day")


def trading_days_left_in_month(index: pd.DatetimeIndex) -> pd.Series:
    """Trading days remaining in the month after each bar (0 = last bar).

    Counts only bars present in ``index``, so holidays and missing sessions
    are excluded automatically.

    Returns:
        Integer Series named ``"days_left"`` aligned to ``index``.
    """
    _validate_index(index)
    n = len(index)
    if n == 0:
        return pd.Series(np.zeros(0, dtype=int), index=index, name="days_left")

    boundary = _period_boundaries(index, "M")
    last = np.append(boundary, True)
    first = np.insert(boundary, 0, True)
    positions = np.arange(n)
    starts = positions[first]
    sizes = np.diff(np.append(starts, n))
    month_end = np.repeat(positions[last], sizes)
    return pd.Series(month_end - positions, index=index, name="days_left")
