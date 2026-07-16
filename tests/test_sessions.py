"""Tests for intraday session filtering."""

import datetime as dt

import pandas as pd
import pytest

from src.data.sessions import filter_session, session_mask


def _minute_frame() -> pd.DataFrame:
    idx = pd.date_range("2024-01-02 00:00", "2024-01-03 23:59", freq="30min")
    return pd.DataFrame({"close": range(len(idx))}, index=idx)


def test_day_session_keeps_only_rth_bars() -> None:
    out = filter_session(_minute_frame(), start="09:30", end="16:00")
    times = pd.DatetimeIndex(out.index)
    assert (times.time >= dt.time(9, 30)).all()
    assert (times.time <= dt.time(16, 0)).all()
    assert len(out) > 0


def test_boundaries_are_inclusive() -> None:
    df = _minute_frame()
    out = filter_session(df, start="09:30", end="16:00")
    kept_times = {ts.time() for ts in out.index}
    assert dt.time(9, 30) in kept_times
    assert dt.time(16, 0) in kept_times
    assert dt.time(9, 0) not in kept_times
    assert dt.time(16, 30) not in kept_times


def test_overnight_session_wraps_midnight() -> None:
    mask = session_mask(pd.DatetimeIndex(_minute_frame().index), start="18:00", end="17:00")
    frame = _minute_frame()[mask.to_numpy()]
    kept = {ts.time() for ts in frame.index}
    assert dt.time(23, 30) in kept  # late evening kept
    assert dt.time(3, 0) in kept  # after midnight kept
    assert dt.time(17, 30) not in kept  # the daily maintenance break dropped


def test_mask_shape_and_name() -> None:
    idx = pd.DatetimeIndex(_minute_frame().index)
    mask = session_mask(idx)
    assert mask.name == "in_session"
    assert len(mask) == len(idx)
    assert mask.dtype == bool


def test_accepts_datetime_time_objects() -> None:
    df = _minute_frame()
    a = filter_session(df, start=dt.time(9, 30), end=dt.time(16, 0))
    b = filter_session(df, start="09:30", end="16:00")
    assert a.index.equals(b.index)


def test_input_is_not_mutated_and_output_is_a_copy() -> None:
    df = _minute_frame()
    out = filter_session(df)
    out.iloc[0, 0] = -1
    assert (df["close"] >= 0).all()


def test_bad_inputs_raise() -> None:
    df = _minute_frame()
    with pytest.raises(ValueError, match="HH:MM"):
        filter_session(df, start="9h30")
    with pytest.raises(ValueError, match="differ"):
        filter_session(df, start="09:30", end="09:30")
    with pytest.raises(TypeError, match="DatetimeIndex"):
        session_mask(pd.Index([1, 2, 3]))  # type: ignore[arg-type]
