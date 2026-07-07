"""Tests for the calendar-seasonality reports."""

import numpy as np
import pandas as pd
import pytest

from src.reporting.seasonality import (
    day_of_week_seasonality,
    monthly_seasonality,
    turn_of_month_effect,
)


def _two_years() -> pd.DatetimeIndex:
    return pd.date_range("2022-01-03", "2023-12-29", freq="B")


def test_monthly_seasonality_compounds_within_month() -> None:
    idx = _two_years()
    values = np.where(idx.month == 1, 0.01, 0.0)
    table = monthly_seasonality(pd.Series(values, index=idx))
    assert table.index.name == "month"
    assert set(table.index) == set(range(1, 13))
    # January 2022 has 21 bdays, January 2023 has 22 -> mean of the two
    # compounded returns, and both are positive
    expected_jan = np.mean([1.01**21 - 1, 1.01**22 - 1])
    assert table.loc[1, "mean_return"] == pytest.approx(expected_jan)
    assert table.loc[1, "hit_rate"] == 1.0
    assert (table.drop(index=1)["mean_return"] == 0.0).all()
    # two years -> two observations per month
    assert (table["n_obs"] == 2).all()


def test_day_of_week_seasonality_isolates_the_weekday() -> None:
    idx = _two_years()
    values = np.where(idx.dayofweek == 0, 0.01, -0.001)
    table = day_of_week_seasonality(pd.Series(values, index=idx))
    assert table.index.name == "day"
    assert list(table.index) == ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
    assert table.loc["Monday", "mean_return"] == pytest.approx(0.01)
    assert table.loc["Monday", "hit_rate"] == 1.0
    assert table.loc["Friday", "mean_return"] == pytest.approx(-0.001)
    assert table.loc["Friday", "hit_rate"] == 0.0


def test_turn_of_month_effect_detects_boundary_returns() -> None:
    idx = _two_years()
    series = pd.Series(0.0, index=idx)
    # mark the first and last 3 trading days of each month as strong
    from src.data.calendar import trading_day_of_month, trading_days_left_in_month

    day_no = trading_day_of_month(idx).to_numpy()
    days_left = trading_days_left_in_month(idx).to_numpy()
    boundary = (day_no <= 3) | (days_left < 3)
    series[boundary] = 0.01

    table = turn_of_month_effect(series, window=3)
    assert list(table.index) == ["turn_of_month", "other"]
    assert table.loc["turn_of_month", "mean_return"] == pytest.approx(0.01)
    assert table.loc["turn_of_month", "hit_rate"] == 1.0
    assert table.loc["other", "mean_return"] == 0.0
    assert int(table["n_obs"].sum()) == len(idx)


def test_turn_of_month_window_widens_the_bucket() -> None:
    idx = _two_years()
    rng = np.random.default_rng(3)
    series = pd.Series(rng.normal(0, 0.01, len(idx)), index=idx)
    narrow = turn_of_month_effect(series, window=1)
    wide = turn_of_month_effect(series, window=5)
    assert wide.loc["turn_of_month", "n_obs"] > narrow.loc["turn_of_month", "n_obs"]


def test_partial_year_reports_only_present_months() -> None:
    idx = pd.date_range("2024-01-02", "2024-03-28", freq="B")
    table = monthly_seasonality(pd.Series(0.001, index=idx))
    assert list(table.index) == [1, 2, 3]


def test_empty_returns_raise() -> None:
    empty = pd.Series(dtype=float, index=pd.DatetimeIndex([]))
    with pytest.raises(ValueError, match="empty"):
        monthly_seasonality(empty)
    with pytest.raises(ValueError, match="empty"):
        day_of_week_seasonality(empty)


def test_non_datetime_index_raises() -> None:
    series = pd.Series([0.01, 0.02])
    with pytest.raises(TypeError, match="DatetimeIndex"):
        monthly_seasonality(series)
    with pytest.raises(TypeError, match="DatetimeIndex"):
        turn_of_month_effect(series)


def test_bad_window_raises() -> None:
    idx = pd.date_range("2024-01-02", periods=30, freq="B")
    with pytest.raises(ValueError, match="window"):
        turn_of_month_effect(pd.Series(0.0, index=idx), window=0)
