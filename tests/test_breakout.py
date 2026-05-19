"""Tests for the Donchian breakout strategy."""

import numpy as np
import pandas as pd

from src.strategy.breakout import breakout_strategy


def _make_ohlc(prices: list[float]) -> pd.DataFrame:
    dates = pd.date_range("2020-01-01", periods=len(prices), freq="B")
    close = np.array(prices, dtype=float)
    return pd.DataFrame(
        {
            "open": close,
            "high": close * 1.005,
            "low": close * 0.995,
            "close": close,
        },
        index=dates,
    )


def test_signal_column_exists(sample_ohlcv):
    out = breakout_strategy(sample_ohlcv, entry_window=20, exit_window=10)
    assert "signal" in out.columns
    assert out["signal"].isin([-1, 0, 1]).all()


def test_long_on_new_high_then_exit_on_pullback():
    # rising series then a sharp pullback that pierces the exit channel
    rising = [100 + i for i in range(30)]
    pullback = [125, 120, 110, 105, 100, 95, 90]
    df = _make_ohlc(rising + pullback)

    out = breakout_strategy(df, entry_window=10, exit_window=5, allow_short=False)

    # somewhere in the rising leg we must be long
    assert (out["signal"].iloc[10:30] == 1).any()
    # by the end of the pullback we must be flat (exit channel triggered)
    assert out["signal"].iloc[-1] == 0


def test_short_disabled_when_allow_short_false():
    # monotonically falling series — would normally trigger shorts
    df = _make_ohlc([100 - i for i in range(40)])
    out = breakout_strategy(df, entry_window=10, exit_window=5, allow_short=False)
    assert (out["signal"] != -1).all()


def test_atr_filter_suppresses_small_breakouts():
    # very gentle drift — breakouts will be tiny compared to ATR
    df = _make_ohlc([100 + 0.001 * i for i in range(50)])
    out_no_filter = breakout_strategy(df, entry_window=10, exit_window=5, atr_filter=0)
    out_with_filter = breakout_strategy(
        df, entry_window=10, exit_window=5, atr_filter=5.0
    )

    # filter must reduce or equal the number of long entries
    assert (out_with_filter["signal"] == 1).sum() <= (out_no_filter["signal"] == 1).sum()


def test_works_without_high_low_columns():
    dates = pd.date_range("2020-01-01", periods=40, freq="B")
    df = pd.DataFrame({"close": [100 + i for i in range(40)]}, index=dates)
    out = breakout_strategy(df, entry_window=10, exit_window=5)
    assert "signal" in out.columns
