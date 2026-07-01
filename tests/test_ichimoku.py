"""Tests for the Ichimoku indicator."""

import numpy as np
import pandas as pd
import pytest

from src.indicators import ichimoku


def _ohlc(close: np.ndarray, spread: float = 0.0) -> pd.DataFrame:
    idx = pd.date_range("2020-01-01", periods=len(close), freq="B")
    c = np.asarray(close, dtype=float)
    return pd.DataFrame({"high": c + spread, "low": c - spread, "close": c}, index=idx)


def test_columns() -> None:
    df = _ohlc(np.arange(1, 120, dtype=float))
    out = ichimoku(df["high"], df["low"], df["close"])
    assert list(out.columns) == ["tenkan", "kijun", "senkou_a", "senkou_b", "chikou"]


def test_tenkan_is_rolling_midpoint() -> None:
    # high==low==close==index -> tenkan[i] = (close[i] + close[i-8]) / 2 = i - 4
    df = _ohlc(np.arange(0, 50, dtype=float))
    out = ichimoku(df["high"], df["low"], df["close"], conversion=9)
    assert out["tenkan"].iloc[10] == pytest.approx(10 - 4)


def test_chikou_is_close_shifted_back() -> None:
    df = _ohlc(np.arange(0, 60, dtype=float))
    out = ichimoku(df["high"], df["low"], df["close"], displacement=26)
    assert out["chikou"].iloc[0] == pytest.approx(df["close"].iloc[26])


def test_senkou_a_is_shifted_forward() -> None:
    df = _ohlc(np.arange(0, 80, dtype=float))
    out = ichimoku(df["high"], df["low"], df["close"], displacement=26)
    base_line = (out["tenkan"] + out["kijun"]) / 2
    # index chosen so the referenced base-line value is past its warm-up (>= 25)
    assert out["senkou_a"].iloc[60] == pytest.approx(base_line.iloc[60 - 26])


def test_tenkan_above_kijun_in_uptrend() -> None:
    df = _ohlc(np.arange(1, 120, dtype=float))
    out = ichimoku(df["high"], df["low"], df["close"]).dropna()
    assert (out["tenkan"] > out["kijun"]).all()


def test_validation() -> None:
    df = _ohlc(np.arange(1, 60, dtype=float))
    with pytest.raises(ValueError, match="span_b"):
        ichimoku(df["high"], df["low"], df["close"], conversion=0)
    with pytest.raises(ValueError, match="displacement"):
        ichimoku(df["high"], df["low"], df["close"], displacement=-1)
