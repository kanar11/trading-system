"""Tests for bar-frequency inference and missing-session detection."""

import pandas as pd
import pytest

from src.data.gaps import gap_report, infer_bar_frequency, missing_sessions


def test_infer_daily_frequency_despite_weekends() -> None:
    idx = pd.date_range("2024-01-01", periods=30, freq="B")
    assert infer_bar_frequency(idx) == pd.Timedelta(days=1)


def test_infer_intraday_frequency() -> None:
    idx = pd.date_range("2024-01-02 09:30", periods=50, freq="30min")
    assert infer_bar_frequency(idx) == pd.Timedelta(minutes=30)


def test_complete_series_has_no_missing_sessions() -> None:
    idx = pd.date_range("2024-01-01", periods=40, freq="B")
    assert len(missing_sessions(idx)) == 0
    assert gap_report(idx).empty


def test_dropped_business_days_are_found() -> None:
    idx = pd.date_range("2024-01-01", periods=40, freq="B")
    holey = idx.delete([5, 6, 20])
    missing = missing_sessions(holey)
    assert list(missing) == [idx[5], idx[6], idx[20]]


def test_weekends_are_never_false_positives() -> None:
    idx = pd.date_range("2024-01-01", periods=40, freq="B")
    missing = missing_sessions(idx, freq="B")
    assert len(missing) == 0  # Saturdays/Sundays are not on the B grid


def test_gap_report_groups_consecutive_and_ranks_by_size() -> None:
    idx = pd.date_range("2024-01-01", periods=60, freq="B")
    holey = idx.delete([10, 11, 12, 30, 45, 46])
    report = gap_report(holey)
    assert list(report["n_missing"]) == [3, 2, 1]  # longest first
    assert report.loc[0, "start"] == idx[10]
    assert report.loc[0, "end"] == idx[12]
    assert report.loc[2, "start"] == idx[30]


def test_intraday_grid_finds_the_lost_hour() -> None:
    idx = pd.date_range("2024-01-02 09:30", periods=14, freq="30min")
    holey = idx.delete([4, 5])  # a lost hour
    report = gap_report(holey, freq="30min")
    assert len(report) == 1
    assert report.loc[0, "n_missing"] == 2


def test_bad_inputs_raise() -> None:
    with pytest.raises(TypeError, match="DatetimeIndex"):
        infer_bar_frequency(pd.Index([1, 2, 3]))  # type: ignore[arg-type]
    unsorted = pd.DatetimeIndex(["2024-01-02", "2024-01-01"])
    with pytest.raises(ValueError, match="sorted"):
        missing_sessions(unsorted)
    dupes = pd.DatetimeIndex(["2024-01-01", "2024-01-01"])
    with pytest.raises(ValueError, match="duplicate"):
        gap_report(dupes)
    single = pd.DatetimeIndex(["2024-01-01"])
    with pytest.raises(ValueError, match="at least 2"):
        infer_bar_frequency(single)
