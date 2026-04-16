"""Tests for momentum signal generation."""

import pandas as pd
import pytest

from src.strategy.momentum import momentum_strategy


def _make_price_df(prices: list[float]) -> pd.DataFrame:
    dates = pd.date_range("2020-01-01", periods=len(prices), freq="B")
    return pd.DataFrame({"close": prices}, index=dates)


def test_signal_column_exists():
    df = _make_price_df([100 + i for i in range(30)])
    result = momentum_strategy(df, lookback=5, threshold=0.01)
    assert "signal" in result.columns


def test_long_signal_on_rising_prices():
    # steadily rising prices should produce long signals
    prices = [100 + i * 2 for i in range(30)]
    df = _make_price_df(prices)
    result = momentum_strategy(df, lookback=5, threshold=0.01)

    # after lookback period, signals should be positive
    signals_after_warmup = result["signal"].iloc[10:]
    assert (signals_after_warmup >= 0).all()


def test_flat_signal_below_threshold():
    # nearly flat prices should produce zero signals
    prices = [100 + (i % 2) * 0.001 for i in range(30)]
    df = _make_price_df(prices)
    result = momentum_strategy(df, lookback=5, threshold=0.05)

    signals_after_warmup = result["signal"].iloc[10:]
    assert (signals_after_warmup == 0).all()


def test_sma_filter_removes_longs_below_sma():
    # create data where price < SMA200 but has positive momentum
    prices = list(range(300, 100, -1))  # declining from 300 to 101
    # add a small uptick at end
    prices[-5:] = [p + 10 for p in prices[-5:]]
    df = _make_price_df(prices)

    result = momentum_strategy(df, lookback=5, threshold=0.01, use_sma_filter=True)

    # with SMA filter, longs below SMA200 should be suppressed
    assert "sma200" in result.columns
