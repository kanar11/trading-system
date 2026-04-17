"""Tests for the mean reversion strategy module."""

import numpy as np
import pandas as pd
import pytest

from src.strategy.mean_reversion import (
    mean_reversion_strategy,
    _bollinger_bands,
    _rsi,
)


def _make_price_df(prices: list[float]) -> pd.DataFrame:
    dates = pd.date_range("2020-01-01", periods=len(prices), freq="B")
    return pd.DataFrame({"close": prices}, index=dates)


class TestBollingerBands:
    def test_returns_three_series(self):
        s = pd.Series([100 + i for i in range(30)])
        middle, upper, lower = _bollinger_bands(s, window=10)

        assert len(middle) == 30
        assert len(upper) == 30
        assert len(lower) == 30

    def test_upper_above_lower(self):
        s = pd.Series([100 + i * 0.5 for i in range(30)])
        middle, upper, lower = _bollinger_bands(s, window=10)

        # after warmup, upper should always be above lower
        valid = middle.dropna().index
        assert (upper[valid] >= lower[valid]).all()

    def test_middle_is_rolling_mean(self):
        s = pd.Series([100 + i for i in range(30)])
        middle, _, _ = _bollinger_bands(s, window=10)
        expected = s.rolling(10).mean()

        pd.testing.assert_series_equal(middle, expected)


class TestRSI:
    def test_rsi_range(self):
        prices = pd.Series([100 + np.sin(i / 3) * 5 for i in range(50)])
        rsi = _rsi(prices, period=14)

        valid = rsi.dropna()
        assert (valid >= 0).all()
        assert (valid <= 100).all()

    def test_rsi_rising_prices(self):
        prices = pd.Series([100 + i for i in range(30)])
        rsi = _rsi(prices, period=14)

        # steadily rising prices should have RSI close to 100
        assert rsi.iloc[-1] > 80


class TestMeanReversionStrategy:
    def test_signal_column_exists(self):
        prices = [100 + np.sin(i / 5) * 10 for i in range(100)]
        df = _make_price_df(prices)
        result = mean_reversion_strategy(df, bb_window=20)

        assert "signal" in result.columns

    def test_indicator_columns_exist(self):
        prices = [100 + np.sin(i / 5) * 10 for i in range(100)]
        df = _make_price_df(prices)
        result = mean_reversion_strategy(df, bb_window=20)

        for col in ["bb_middle", "bb_upper", "bb_lower", "rsi", "percent_b"]:
            assert col in result.columns, f"Missing column: {col}"

    def test_signals_are_valid_values(self):
        prices = [100 + np.sin(i / 3) * 15 for i in range(100)]
        df = _make_price_df(prices)
        result = mean_reversion_strategy(df, bb_window=20)

        unique = set(result["signal"].unique())
        assert unique.issubset({-1, 0, 1})

    def test_no_rsi_filter(self):
        prices = [100 + np.sin(i / 3) * 15 for i in range(100)]
        df = _make_price_df(prices)
        result = mean_reversion_strategy(df, bb_window=20, use_rsi_filter=False)

        assert "signal" in result.columns
