"""Intraday session filtering (time-of-day windows).

Intraday feeds routinely include pre-market, after-hours and overnight
bars whose thin liquidity poisons indicators and backtests calibrated on
regular trading hours. These helpers cut a bar frame down to a
time-of-day window, handling both day sessions (09:30-16:00 US equities)
and *overnight* sessions that wrap midnight (18:00-17:00 CME globex
style): when ``start > end`` the window is interpreted as wrapping.

Both boundaries are inclusive. Time-of-day only — date gaps, holidays
and weekends are whatever the input index contains (see
:mod:`src.data.calendar` for calendar logic). Direct-import module::

    from src.data.sessions import filter_session, session_mask
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd


def _parse_time(value: str | dt.time, name: str) -> dt.time:
    if isinstance(value, dt.time):
        return value
    try:
        return dt.time.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be a 'HH:MM[:SS]' string or datetime.time.") from exc


def session_mask(
    index: pd.DatetimeIndex,
    start: str | dt.time = "09:30",
    end: str | dt.time = "16:00",
) -> pd.Series:
    """Boolean mask of bars whose time-of-day falls inside the session.

    Args:
        index: Bar timestamps.
        start: Session open (inclusive), ``"HH:MM"`` or ``datetime.time``.
        end: Session close (inclusive). ``start > end`` means an overnight
            session wrapping midnight.

    Returns:
        Boolean Series named ``"in_session"`` aligned to ``index``.

    Raises:
        TypeError: If ``index`` is not a DatetimeIndex.
        ValueError: If a time cannot be parsed or ``start == end``.
    """
    if not isinstance(index, pd.DatetimeIndex):
        raise TypeError(f"index must be a DatetimeIndex, got {type(index).__name__}.")
    open_time = _parse_time(start, "start")
    close_time = _parse_time(end, "end")
    if open_time == close_time:
        raise ValueError("start and end must differ.")

    times = np.array([ts.time() for ts in index])
    if open_time < close_time:
        mask = (times >= open_time) & (times <= close_time)
    else:  # overnight session wrapping midnight
        mask = (times >= open_time) | (times <= close_time)
    return pd.Series(mask, index=index, name="in_session")


def filter_session(
    df: pd.DataFrame,
    start: str | dt.time = "09:30",
    end: str | dt.time = "16:00",
) -> pd.DataFrame:
    """Return only the rows of ``df`` inside the session window.

    See :func:`session_mask` for the window semantics; the result is a
    copy, the input is not mutated.

    Raises:
        TypeError: If the frame's index is not a DatetimeIndex.
        ValueError: If a time cannot be parsed or ``start == end``.
    """
    mask = session_mask(pd.DatetimeIndex(df.index), start=start, end=end)
    return df.loc[mask.to_numpy()].copy()
