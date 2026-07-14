"""Tests for the turn-of-month seasonal strategy."""

import numpy as np
import pandas as pd
import pytest

from src.backtest import backtest_strategy
from src.data.calendar import trading_day_of_month, trading_days_left_in_month
from src.strategy.turn_of_month import turn_of_month_strategy


def _frame(n: int = 130) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=n, freq="B")
    return pd.DataFrame({"close": 100.0 + np.arange(n, dtype=float)}, index=idx)


def test_classic_window_matches_calendar_helpers() -> None:
    df = _frame()
    out = turn_of_month_strategy(df, days_before=1, days_after=3)
    idx = pd.DatetimeIndex(df.index)
    expected = (trading_days_left_in_month(idx).to_numpy() < 1) | (
        trading_day_of_month(idx).to_numpy() <= 3
    )
    assert np.array_equal(out["signal"].to_numpy() == 1, expected)
    assert set(out["signal"].unique()) <= {0, 1}


def test_first_bars_of_month_are_long() -> None:
    out = turn_of_month_strategy(_frame(), days_before=1, days_after=3)
    # 2024-01-01..03 are the first three trading days of January
    assert (out["signal"].iloc[:3] == 1).all()
    # mid-month bar is flat
    assert out["signal"].loc["2024-01-15"] == 0


def test_last_day_of_month_is_long() -> None:
    out = turn_of_month_strategy(_frame(), days_before=1, days_after=3)
    assert out["signal"].loc["2024-01-31"] == 1
    assert out["signal"].loc["2024-01-30"] == 0  # only the final day with days_before=1


def test_days_before_zero_keeps_only_month_start() -> None:
    out = turn_of_month_strategy(_frame(), days_before=0, days_after=2)
    assert out["signal"].loc["2024-01-31"] == 0
    assert out["signal"].loc["2024-02-01"] == 1
    assert out["signal"].loc["2024-02-02"] == 1
    assert out["signal"].loc["2024-02-05"] == 0


def test_wider_window_holds_more_bars() -> None:
    df = _frame()
    narrow = turn_of_month_strategy(df, days_before=1, days_after=1)
    wide = turn_of_month_strategy(df, days_before=3, days_after=5)
    assert int(wide["signal"].sum()) > int(narrow["signal"].sum())


def test_runs_through_the_backtest_engine() -> None:
    out = turn_of_month_strategy(_frame())
    result, _ = backtest_strategy(out, transaction_cost=0.0005)
    assert np.isfinite(result["equity_curve"].to_numpy()).all()


def test_bad_inputs_raise() -> None:
    df = _frame()
    with pytest.raises(ValueError, match="close"):
        turn_of_month_strategy(df.drop(columns=["close"]))
    with pytest.raises(ValueError, match=">= 0"):
        turn_of_month_strategy(df, days_before=-1)
    with pytest.raises(ValueError, match="at least one"):
        turn_of_month_strategy(df, days_before=0, days_after=0)
