"""Tests for the trading-calendar helpers."""

import numpy as np
import pandas as pd
import pytest

from src.data.calendar import (
    rebalance_dates,
    rebalance_mask,
    trading_day_of_month,
    trading_days_left_in_month,
)


def _bdays(start: str = "2024-01-01", periods: int = 130) -> pd.DatetimeIndex:
    return pd.date_range(start, periods=periods, freq="B")


def test_month_end_mask_marks_last_bday_of_each_month() -> None:
    idx = _bdays()  # Jan..late Jun 2024
    mask = rebalance_mask(idx, freq="M", which="last")
    assert mask.name == "rebalance"
    assert mask.dtype == bool
    chosen = idx[mask.to_numpy()]
    # 2024 last business days: Jan 31, Feb 29 (leap), Mar 29 (Good Friday
    # is not a bday gap for freq="B"; the last B-day of March 2024 is Fri 29)
    assert pd.Timestamp("2024-01-31") in chosen
    assert pd.Timestamp("2024-02-29") in chosen
    assert pd.Timestamp("2024-03-29") in chosen
    # exactly one mark per month present in the index
    assert int(mask.sum()) == len(idx.to_period("M").unique())


def test_month_start_dates() -> None:
    idx = _bdays()
    dates = rebalance_dates(idx, freq="M", which="first")
    assert dates[0] == idx[0]
    assert pd.Timestamp("2024-02-01") in dates
    assert pd.Timestamp("2024-04-01") in dates


def test_weekly_mask_one_per_week() -> None:
    idx = _bdays(periods=25)  # five full weeks
    mask = rebalance_mask(idx, freq="W", which="last")
    assert int(mask.sum()) == len(idx.to_period("W").unique())
    # last bar of a full business week is a Friday
    fridays = idx[mask.to_numpy()][:-1]
    assert all(ts.dayofweek == 4 for ts in fridays)


def test_quarterly_and_yearly_counts() -> None:
    idx = pd.date_range("2022-01-03", "2023-12-29", freq="B")
    assert len(rebalance_dates(idx, freq="Q")) == 8
    assert len(rebalance_dates(idx, freq="Y")) == 2


def test_first_and_last_masks_partition_boundaries() -> None:
    idx = _bdays()
    first = rebalance_mask(idx, freq="M", which="first")
    last = rebalance_mask(idx, freq="M", which="last")
    assert first.iloc[0]
    assert last.iloc[-1]
    assert int(first.sum()) == int(last.sum())


def test_trading_day_of_month_counts_from_one() -> None:
    idx = _bdays()
    day = trading_day_of_month(idx)
    assert day.iloc[0] == 1
    # January 2024 has 23 business days
    jan = day[idx.to_period("M") == pd.Period("2024-01", "M")]
    assert list(jan) == list(range(1, 24))


def test_trading_days_left_hits_zero_at_month_end() -> None:
    idx = _bdays()
    left = trading_days_left_in_month(idx)
    month_ends = rebalance_mask(idx, freq="M", which="last")
    assert (left[month_ends.to_numpy()] == 0).all()
    assert left.iloc[0] == 22  # 23 bdays in Jan 2024 -> 22 remain after day 1


def test_day_of_month_plus_days_left_is_month_length() -> None:
    idx = _bdays()
    total = trading_day_of_month(idx) + trading_days_left_in_month(idx)
    # constant within each month: length of that month in bars
    for _, group in total.groupby(idx.to_period("M")):
        assert group.nunique() == 1


def test_empty_index_returns_empty() -> None:
    idx = pd.DatetimeIndex([])
    assert len(rebalance_mask(idx)) == 0
    assert len(trading_day_of_month(idx)) == 0
    assert len(trading_days_left_in_month(idx)) == 0


def test_unsorted_index_raises() -> None:
    idx = pd.DatetimeIndex(["2024-01-02", "2024-01-01"])
    with pytest.raises(ValueError, match="sorted"):
        rebalance_mask(idx)


def test_duplicate_index_raises() -> None:
    idx = pd.DatetimeIndex(["2024-01-01", "2024-01-01"])
    with pytest.raises(ValueError, match="duplicate"):
        rebalance_mask(idx)


def test_non_datetime_index_raises() -> None:
    with pytest.raises(TypeError, match="DatetimeIndex"):
        rebalance_mask(pd.Index(np.arange(5)))  # type: ignore[arg-type]


def test_unknown_freq_or_which_raises() -> None:
    idx = _bdays(periods=10)
    with pytest.raises(ValueError, match="freq"):
        rebalance_mask(idx, freq="D")
    with pytest.raises(ValueError, match="which"):
        rebalance_mask(idx, which="middle")
