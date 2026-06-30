"""Tests for the second-pass indicators (SuperTrend, Vortex)."""

import numpy as np
import pandas as pd
import pytest

from src.indicators import supertrend, vortex


def _ohlc(close: np.ndarray, spread: float = 1.0) -> pd.DataFrame:
    idx = pd.date_range("2020-01-01", periods=len(close), freq="B")
    c = np.asarray(close, dtype=float)
    return pd.DataFrame({"high": c + spread, "low": c - spread, "close": c}, index=idx)


# --- SuperTrend ------------------------------------------------------------


def test_supertrend_columns_and_domain(sample_ohlcv: pd.DataFrame) -> None:
    out = supertrend(sample_ohlcv["high"], sample_ohlcv["low"], sample_ohlcv["close"], period=10)
    assert list(out.columns) == ["supertrend", "direction"]
    assert set(out["direction"].unique()).issubset({-1, 0, 1})


def test_supertrend_line_brackets_price() -> None:
    df = _ohlc(np.arange(1, 61, dtype=float))
    out = supertrend(df["high"], df["low"], df["close"], period=10).dropna()
    close = df["close"].reindex(out.index)
    up, dn = out["direction"] == 1, out["direction"] == -1
    # uptrend line sits below price; downtrend line sits above it
    assert (out["supertrend"][up] <= close[up] + 1e-9).all()
    assert (out["supertrend"][dn] >= close[dn] - 1e-9).all()


def test_supertrend_uptrend() -> None:
    df = _ohlc(np.arange(1, 61, dtype=float))
    out = supertrend(df["high"], df["low"], df["close"], period=10)
    assert out["direction"].iloc[-1] == 1
    assert out["supertrend"].iloc[-1] < df["close"].iloc[-1]


def test_supertrend_downtrend() -> None:
    df = _ohlc(np.arange(60, 0, -1, dtype=float))
    out = supertrend(df["high"], df["low"], df["close"], period=10)
    assert out["direction"].iloc[-1] == -1
    assert out["supertrend"].iloc[-1] > df["close"].iloc[-1]


def test_supertrend_bad_period() -> None:
    df = _ohlc(np.arange(1, 40, dtype=float))
    with pytest.raises(ValueError, match="period"):
        supertrend(df["high"], df["low"], df["close"], period=0)


# --- Vortex ----------------------------------------------------------------


def test_vortex_columns_and_nonnegative(sample_ohlcv: pd.DataFrame) -> None:
    out = vortex(
        sample_ohlcv["high"], sample_ohlcv["low"], sample_ohlcv["close"], period=14
    ).dropna()
    assert list(out.columns) == ["vi_plus", "vi_minus"]
    assert (out["vi_plus"] >= 0).all()
    assert (out["vi_minus"] >= 0).all()


def test_vortex_uptrend_plus_dominates() -> None:
    df = _ohlc(np.arange(1, 61, dtype=float))
    out = vortex(df["high"], df["low"], df["close"], period=14)
    assert out["vi_plus"].iloc[-1] > out["vi_minus"].iloc[-1]


def test_vortex_downtrend_minus_dominates() -> None:
    df = _ohlc(np.arange(60, 0, -1, dtype=float))
    out = vortex(df["high"], df["low"], df["close"], period=14)
    assert out["vi_minus"].iloc[-1] > out["vi_plus"].iloc[-1]


def test_vortex_bad_period() -> None:
    df = _ohlc(np.arange(1, 40, dtype=float))
    with pytest.raises(ValueError, match="period"):
        vortex(df["high"], df["low"], df["close"], period=0)
