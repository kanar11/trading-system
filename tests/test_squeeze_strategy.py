"""Tests for the Bollinger/Keltner squeeze strategy."""

import numpy as np
import pandas as pd
import pytest

from src.strategy.squeeze import squeeze_strategy


def _ohlc(close: np.ndarray, spread: float = 0.0) -> pd.DataFrame:
    idx = pd.date_range("2020-01-01", periods=len(close), freq="B")
    c = np.asarray(close, dtype=float)
    return pd.DataFrame({"high": c + spread, "low": c - spread, "close": c}, index=idx)


def test_outputs_and_signal_domain(sample_ohlcv: pd.DataFrame) -> None:
    out = squeeze_strategy(sample_ohlcv)
    assert "squeeze_on" in out.columns
    assert "signal" in out.columns
    assert set(out["signal"].unique()).issubset({-1, 0, 1})


def test_flat_while_squeezed(sample_ohlcv: pd.DataFrame) -> None:
    out = squeeze_strategy(sample_ohlcv)
    squeezed = out[out["squeeze_on"]]
    assert (squeezed["signal"] == 0).all()


def test_long_in_uptrend_when_not_squeezed() -> None:
    # tiny Keltner mult -> never squeezed; rising price -> positive momentum -> long
    out = squeeze_strategy(_ohlc(np.arange(1, 80, dtype=float)), kc_atr_mult=0.1)
    assert out["signal"].iloc[-1] == 1


def test_short_in_downtrend_when_not_squeezed() -> None:
    out = squeeze_strategy(_ohlc(np.arange(80, 1, -1, dtype=float)), kc_atr_mult=0.1)
    assert out["signal"].iloc[-1] == -1


def test_allow_short_false_clamps() -> None:
    out = squeeze_strategy(
        _ohlc(np.arange(80, 1, -1, dtype=float)), kc_atr_mult=0.1, allow_short=False
    )
    assert (out["signal"] >= 0).all()


def test_does_not_mutate_input(sample_ohlcv: pd.DataFrame) -> None:
    before = set(sample_ohlcv.columns)
    squeeze_strategy(sample_ohlcv)
    assert set(sample_ohlcv.columns) == before


def test_missing_columns_raise() -> None:
    with pytest.raises(ValueError, match="must contain columns"):
        squeeze_strategy(pd.DataFrame({"close": [1.0, 2.0, 3.0]}))
