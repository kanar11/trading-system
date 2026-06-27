"""Tests for the additional indicators (HMA, Aroon, TRIX, CMO, MFI)."""

import numpy as np
import pandas as pd
import pytest

from src.indicators import aroon, cmo, hma, mfi, trix

# --- Hull Moving Average ---------------------------------------------------


def test_hma_tracks_linear_trend_monotonically() -> None:
    close = pd.Series(np.arange(100, dtype=float))
    out = hma(close, 16).dropna()
    assert (out.diff().dropna() > 0).all()


def test_hma_rejects_tiny_window() -> None:
    with pytest.raises(ValueError, match="window"):
        hma(pd.Series([1.0, 2.0, 3.0]), 1)


# --- Aroon -----------------------------------------------------------------


def test_aroon_columns_and_range(sample_ohlcv: pd.DataFrame) -> None:
    out = aroon(sample_ohlcv["high"], sample_ohlcv["low"], period=25).dropna()
    assert list(out.columns) == ["up", "down", "oscillator"]
    assert ((out["up"] >= 0) & (out["up"] <= 100)).all()
    assert ((out["down"] >= 0) & (out["down"] <= 100)).all()


def test_aroon_saturates_on_monotonic_series() -> None:
    high = pd.Series(np.arange(1, 61, dtype=float))
    low = pd.Series(np.arange(1, 61, dtype=float))
    out = aroon(high, low, period=10)
    assert out["up"].iloc[-1] == 100.0
    assert out["down"].iloc[-1] == 0.0
    assert out["oscillator"].iloc[-1] == 100.0


# --- TRIX ------------------------------------------------------------------


def test_trix_zero_on_constant_series() -> None:
    out = trix(pd.Series(np.full(100, 5.0)), period=10)
    assert abs(out.iloc[-1]) < 1e-9


def test_trix_positive_in_uptrend() -> None:
    out = trix(pd.Series(np.arange(1, 101, dtype=float)), period=10)
    assert out.iloc[-1] > 0


def test_trix_rejects_bad_period() -> None:
    with pytest.raises(ValueError, match="period"):
        trix(pd.Series([1.0, 2.0, 3.0]), period=0)


# --- Chande Momentum Oscillator -------------------------------------------


def test_cmo_extremes_on_monotonic_series() -> None:
    up = cmo(pd.Series(np.arange(1, 60, dtype=float)), period=14)
    down = cmo(pd.Series(np.arange(60, 1, -1, dtype=float)), period=14)
    assert up.iloc[-1] == pytest.approx(100.0)
    assert down.iloc[-1] == pytest.approx(-100.0)


def test_cmo_within_bounds(sample_ohlcv: pd.DataFrame) -> None:
    out = cmo(sample_ohlcv["close"], period=14).dropna()
    assert ((out >= -100) & (out <= 100)).all()


# --- Money Flow Index ------------------------------------------------------


def test_mfi_within_bounds(sample_ohlcv: pd.DataFrame) -> None:
    out = mfi(
        sample_ohlcv["high"],
        sample_ohlcv["low"],
        sample_ohlcv["close"],
        sample_ohlcv["volume"],
        period=14,
    ).dropna()
    assert ((out >= 0) & (out <= 100)).all()


def test_mfi_saturates_when_only_positive_flow() -> None:
    rising = pd.Series(np.arange(1, 61, dtype=float))
    vol = pd.Series(np.full(60, 1000.0))
    out = mfi(rising, rising, rising, vol, period=14)
    assert out.iloc[-1] == pytest.approx(100.0)


def test_mfi_rejects_bad_period() -> None:
    s = pd.Series(np.arange(1, 60, dtype=float))
    with pytest.raises(ValueError, match="period"):
        mfi(s, s, s, s, period=0)
