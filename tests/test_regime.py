"""Tests for the regime detection module."""

import numpy as np
import pandas as pd
import pytest

from src.regime.detector import (
    RegimeConfig,
    RegimeType,
    detect_regime,
    adaptive_strategy,
    _adx,
    _rolling_hurst,
)
from src.strategy.momentum import momentum_strategy
from src.strategy.mean_reversion import mean_reversion_strategy
from functools import partial


def _make_ohlc_df(n: int = 300, trend: bool = True) -> pd.DataFrame:
    """Generate synthetic OHLC data."""
    np.random.seed(42)
    dates = pd.date_range("2018-01-01", periods=n, freq="B")

    if trend:
        # strong uptrend with noise
        close = 100 + np.cumsum(np.random.normal(0.2, 0.5, n))
    else:
        # mean-reverting around 100
        close = 100 + np.cumsum(np.random.normal(0, 1, n))
        close = 100 + (close - close.mean()) * 0.3  # squash

    high = close + np.abs(np.random.normal(0.5, 0.3, n))
    low = close - np.abs(np.random.normal(0.5, 0.3, n))

    return pd.DataFrame(
        {"open": close, "high": high, "low": low, "close": close},
        index=dates,
    )


def _make_close_only_df(n: int = 300) -> pd.DataFrame:
    np.random.seed(42)
    dates = pd.date_range("2018-01-01", periods=n, freq="B")
    close = 100 + np.cumsum(np.random.normal(0.1, 0.8, n))
    return pd.DataFrame({"close": close}, index=dates)


class TestADX:
    def test_returns_series(self):
        df = _make_ohlc_df(200)
        adx = _adx(df["high"], df["low"], df["close"], period=14)
        assert isinstance(adx, pd.Series)
        assert len(adx) == 200

    def test_adx_non_negative(self):
        df = _make_ohlc_df(200)
        adx = _adx(df["high"], df["low"], df["close"], period=14)
        valid = adx.dropna()
        assert (valid >= 0).all()


class TestHurst:
    def test_returns_series(self):
        df = _make_close_only_df(200)
        h = _rolling_hurst(df["close"], window=50)
        assert isinstance(h, pd.Series)
        assert len(h) == 200

    def test_hurst_in_valid_range(self):
        df = _make_close_only_df(300)
        h = _rolling_hurst(df["close"], window=50)
        valid = h.dropna()
        # Hurst should be roughly between 0 and 1
        assert (valid > -0.5).all()
        assert (valid < 1.5).all()


class TestDetectRegime:
    def test_adds_regime_column(self):
        df = _make_ohlc_df(300)
        result = detect_regime(df)
        assert "regime" in result.columns
        assert "adx" in result.columns
        assert "hurst" in result.columns
        assert "vol_regime" in result.columns

    def test_regime_values_are_valid(self):
        df = _make_ohlc_df(300)
        result = detect_regime(df)
        valid_types = {RegimeType.TRENDING, RegimeType.MEAN_REVERTING, RegimeType.UNDEFINED}
        assert set(result["regime"].unique()).issubset(valid_types)

    def test_works_without_high_low(self):
        df = _make_close_only_df(300)
        result = detect_regime(df)
        assert "regime" in result.columns
        assert "adx" in result.columns

    def test_custom_config(self):
        df = _make_ohlc_df(300)
        config = RegimeConfig(
            adx_period=10,
            adx_trending_threshold=20.0,
            hurst_window=50,
            smoothing_window=3,
        )
        result = detect_regime(df, config=config)
        assert "regime" in result.columns

    def test_smoothing_reduces_changes(self):
        df = _make_ohlc_df(300)
        no_smooth = detect_regime(df, RegimeConfig(smoothing_window=1))
        smoothed = detect_regime(df, RegimeConfig(smoothing_window=10))

        changes_raw = (no_smooth["regime"] != no_smooth["regime"].shift(1)).sum()
        changes_smooth = (smoothed["regime"] != smoothed["regime"].shift(1)).sum()

        # smoothing should reduce or equal the number of regime changes
        assert changes_smooth <= changes_raw


class TestAdaptiveStrategy:
    def test_returns_signal_column(self):
        df = _make_ohlc_df(300)
        mom_fn = partial(momentum_strategy, lookback=20, threshold=0.01)
        mr_fn = partial(mean_reversion_strategy, bb_window=20)

        result = adaptive_strategy(df, mom_fn, mr_fn)
        assert "signal" in result.columns
        assert "regime" in result.columns

    def test_signals_are_valid(self):
        df = _make_ohlc_df(300)
        mom_fn = partial(momentum_strategy, lookback=20, threshold=0.01)
        mr_fn = partial(mean_reversion_strategy, bb_window=20)

        result = adaptive_strategy(df, mom_fn, mr_fn)
        assert set(result["signal"].unique()).issubset({-1, 0, 1})

    def test_flat_during_undefined(self):
        df = _make_ohlc_df(300)
        mom_fn = partial(momentum_strategy, lookback=20, threshold=0.01)
        mr_fn = partial(mean_reversion_strategy, bb_window=20)

        result = adaptive_strategy(df, mom_fn, mr_fn)
        undefined_signals = result.loc[result["regime"] == RegimeType.UNDEFINED, "signal"]

        # all signals during undefined regime should be 0
        assert (undefined_signals == 0).all()
