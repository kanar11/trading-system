"""Tests for Elder-Ray and Chaikin Volatility indicators."""

import numpy as np
import pandas as pd
import pytest

from src.indicators import chaikin_volatility, elder_ray
from src.indicators.trend import ema


def _ohlc(close: np.ndarray, spread: float = 1.0) -> pd.DataFrame:
    idx = pd.date_range("2020-01-01", periods=len(close), freq="B")
    c = np.asarray(close, dtype=float)
    return pd.DataFrame({"high": c + spread, "low": c - spread, "close": c}, index=idx)


# --- Elder-Ray -------------------------------------------------------------


def test_elder_ray_columns_and_formula() -> None:
    df = _ohlc(np.arange(1, 60, dtype=float))
    out = elder_ray(df["high"], df["low"], df["close"], period=13)
    assert list(out.columns) == ["bull_power", "bear_power"]
    baseline = ema(df["close"], 13)
    assert out["bull_power"].dropna().equals((df["high"] - baseline).dropna())


def test_elder_ray_bull_exceeds_bear() -> None:
    df = _ohlc(np.arange(1, 60, dtype=float), spread=1.0)
    out = elder_ray(df["high"], df["low"], df["close"]).dropna()
    # bull - bear == high - low == 2*spread > 0
    assert (out["bull_power"] > out["bear_power"]).all()


def test_elder_ray_positive_bull_in_uptrend() -> None:
    df = _ohlc(np.arange(1, 80, dtype=float))
    out = elder_ray(df["high"], df["low"], df["close"])
    assert out["bull_power"].iloc[-1] > 0


def test_elder_ray_bad_period() -> None:
    df = _ohlc(np.arange(1, 30, dtype=float))
    with pytest.raises(ValueError, match="period"):
        elder_ray(df["high"], df["low"], df["close"], period=0)


# --- Chaikin Volatility ----------------------------------------------------


def test_chaikin_volatility_positive_when_range_expands() -> None:
    # widening high-low range -> positive Chaikin volatility
    n = 60
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    close = pd.Series(100.0, index=idx)
    rng = np.arange(1, n + 1, dtype=float)  # expanding range
    high = close + rng
    low = close - rng
    out = chaikin_volatility(high, low, period=10).dropna()
    assert (out > 0).all()


def test_chaikin_volatility_negative_when_range_contracts() -> None:
    n = 60
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    close = pd.Series(100.0, index=idx)
    rng = np.arange(n, 0, -1, dtype=float)  # contracting range
    out = chaikin_volatility(close + rng, close - rng, period=10).dropna()
    assert (out < 0).all()


def test_chaikin_volatility_bad_period() -> None:
    s = pd.Series(np.arange(1, 30, dtype=float))
    with pytest.raises(ValueError, match="period"):
        chaikin_volatility(s + 1, s - 1, period=0)
