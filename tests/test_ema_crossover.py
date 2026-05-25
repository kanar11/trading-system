"""Tests for EMA and MACD crossover strategies."""

import numpy as np
import pandas as pd
import pytest

from src.strategy.ema_crossover import ema_crossover_strategy, macd_strategy


def _df_from_prices(prices: list[float]) -> pd.DataFrame:
    dates = pd.date_range("2020-01-01", periods=len(prices), freq="B")
    return pd.DataFrame({"close": prices}, index=dates)


# ---------------------------------------------------------------------------
# EMA crossover
# ---------------------------------------------------------------------------

def test_ema_long_on_rising_series():
    df = _df_from_prices([100 + i for i in range(60)])
    out = ema_crossover_strategy(df, fast=5, slow=20)
    # after warm-up, fast EMA leads slow EMA upward → long
    assert (out["signal"].iloc[25:] == 1).all()


def test_ema_short_on_falling_series():
    df = _df_from_prices([200 - i for i in range(60)])
    out = ema_crossover_strategy(df, fast=5, slow=20)
    assert (out["signal"].iloc[25:] == -1).all()


def test_ema_short_disabled_clamps_to_flat():
    df = _df_from_prices([200 - i for i in range(60)])
    out = ema_crossover_strategy(df, fast=5, slow=20, allow_short=False)
    assert (out["signal"] != -1).all()


def test_ema_gap_suppresses_tiny_crossovers():
    # nearly flat noise
    np.random.seed(0)
    prices = list(100 + np.random.normal(0, 0.05, 100))
    df = _df_from_prices(prices)
    no_gap = ema_crossover_strategy(df, fast=5, slow=20, gap_bps=0)
    big_gap = ema_crossover_strategy(df, fast=5, slow=20, gap_bps=1000)
    assert (big_gap["signal"] != 0).sum() <= (no_gap["signal"] != 0).sum()


def test_ema_fast_ge_slow_raises():
    df = _df_from_prices([100] * 50)
    with pytest.raises(ValueError, match="fast"):
        ema_crossover_strategy(df, fast=20, slow=10)


# ---------------------------------------------------------------------------
# MACD
# ---------------------------------------------------------------------------

def test_macd_columns_present():
    df = _df_from_prices([100 + i for i in range(80)])
    out = macd_strategy(df)
    for col in ("macd", "macd_signal", "macd_hist", "signal"):
        assert col in out.columns


def test_macd_hist_long_short_consistency():
    df = _df_from_prices([100 + i for i in range(80)])
    out = macd_strategy(df, fast=12, slow=26, signal_span=9)
    # whenever histogram is positive and signal is set, it must be +1
    pos_mask = out["macd_hist"] > 0
    assert (out.loc[pos_mask, "signal"] == 1).all()


def test_macd_short_disabled_clamps_to_flat():
    df = _df_from_prices([200 - i for i in range(80)])
    out = macd_strategy(df, allow_short=False)
    assert (out["signal"] != -1).all()


def test_macd_fast_ge_slow_raises():
    df = _df_from_prices([100] * 50)
    with pytest.raises(ValueError, match="fast"):
        macd_strategy(df, fast=26, slow=12)
