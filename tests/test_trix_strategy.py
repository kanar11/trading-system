"""Tests for the TRIX trend-following strategy."""

import numpy as np
import pandas as pd
import pytest

from src.strategy.trix import trix_strategy


def _frame(close: np.ndarray) -> pd.DataFrame:
    idx = pd.date_range("2020-01-01", periods=len(close), freq="B")
    return pd.DataFrame({"close": close}, index=idx)


def test_outputs_columns_and_signal_domain(sample_ohlcv: pd.DataFrame) -> None:
    out = trix_strategy(sample_ohlcv, period=15)
    assert "trix" in out.columns
    assert "signal" in out.columns
    assert set(out["signal"].unique()).issubset({-1, 0, 1})


def test_long_in_uptrend() -> None:
    out = trix_strategy(_frame(np.arange(1, 201, dtype=float)), period=15)
    assert out["signal"].iloc[-1] == 1


def test_short_in_downtrend() -> None:
    out = trix_strategy(_frame(np.arange(200, 0, -1, dtype=float)), period=15)
    assert out["signal"].iloc[-1] == -1


def test_allow_short_false_clamps_shorts() -> None:
    out = trix_strategy(_frame(np.arange(200, 0, -1, dtype=float)), period=15, allow_short=False)
    assert (out["signal"] >= 0).all()


def test_signal_line_mode_adds_column(sample_ohlcv: pd.DataFrame) -> None:
    out = trix_strategy(sample_ohlcv, period=15, signal_period=9, use_signal_line=True)
    assert "trix_signal" in out.columns
    assert set(out["signal"].unique()).issubset({-1, 0, 1})


def test_does_not_mutate_input(sample_ohlcv: pd.DataFrame) -> None:
    before = set(sample_ohlcv.columns)
    trix_strategy(sample_ohlcv, period=15)
    assert set(sample_ohlcv.columns) == before


def test_missing_close_raises() -> None:
    with pytest.raises(ValueError, match="close"):
        trix_strategy(pd.DataFrame({"open": [1.0, 2.0, 3.0]}))


def test_bad_period_raises() -> None:
    with pytest.raises(ValueError, match="period"):
        trix_strategy(_frame(np.arange(1, 50, dtype=float)), period=0)


def test_bad_signal_period_raises() -> None:
    with pytest.raises(ValueError, match="signal_period"):
        trix_strategy(_frame(np.arange(1, 50, dtype=float)), signal_period=0, use_signal_line=True)
