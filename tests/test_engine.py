"""Tests for the backtest engine."""

import pandas as pd
import pytest

from src.backtest.engine import backtest_strategy


def _make_signal_df(prices: list[float], signals: list[int]) -> pd.DataFrame:
    dates = pd.date_range("2020-01-01", periods=len(prices), freq="B")
    return pd.DataFrame({"close": prices, "signal": signals}, index=dates)


def test_backtest_returns_dataframe_and_trade_log():
    df = _make_signal_df([100, 101, 102, 103, 104], [0, 1, 1, 1, 0])
    result_df, trade_log = backtest_strategy(df, transaction_cost=0.001)

    assert "equity_curve" in result_df.columns
    assert "strategy_returns" in result_df.columns
    assert isinstance(trade_log, pd.DataFrame)


def test_backtest_equity_starts_at_one():
    df = _make_signal_df([100, 101, 102, 103, 104], [0, 0, 0, 0, 0])
    result_df, _ = backtest_strategy(df)

    # with all-zero signals, equity should stay at 1.0
    assert abs(result_df["equity_curve"].iloc[-1] - 1.0) < 1e-10


def test_backtest_raises_without_signal():
    dates = pd.date_range("2020-01-01", periods=3, freq="B")
    df = pd.DataFrame({"close": [100, 101, 102]}, index=dates)

    with pytest.raises(ValueError, match="signal"):
        backtest_strategy(df)


def test_backtest_raises_without_close():
    dates = pd.date_range("2020-01-01", periods=3, freq="B")
    df = pd.DataFrame({"signal": [0, 1, 1]}, index=dates)

    with pytest.raises(ValueError, match="close"):
        backtest_strategy(df)


def test_backtest_with_vol_target():
    prices = [100 + i for i in range(50)]
    signals = [0] * 5 + [1] * 40 + [0] * 5
    df = _make_signal_df(prices, signals)

    result_df, _ = backtest_strategy(df, vol_target=0.15, vol_window=10)

    assert "realized_vol" in result_df.columns
    assert "vol_scalar" in result_df.columns
    assert "scaled_position" in result_df.columns
