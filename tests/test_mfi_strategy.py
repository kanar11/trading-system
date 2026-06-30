"""Tests for the Money Flow Index mean-reversion strategy."""

import numpy as np
import pandas as pd
import pytest

from src.strategy.mfi import mfi_strategy


def _frame(values: np.ndarray) -> pd.DataFrame:
    idx = pd.date_range("2020-01-01", periods=len(values), freq="B")
    v = np.asarray(values, dtype=float)
    return pd.DataFrame(
        {"high": v, "low": v, "close": v, "volume": np.full(len(v), 1000.0)}, index=idx
    )


def test_outputs_columns_and_signal_domain(sample_ohlcv: pd.DataFrame) -> None:
    out = mfi_strategy(sample_ohlcv, period=14)
    assert "mfi" in out.columns
    assert "signal" in out.columns
    assert set(out["signal"].unique()).issubset({-1, 0, 1})


def test_long_when_money_flow_washed_out() -> None:
    # strictly falling -> MFI collapses to 0 -> below oversold -> long
    out = mfi_strategy(_frame(np.arange(60, 0, -1, dtype=float)), period=14)
    assert out["signal"].iloc[-1] == 1


def test_short_when_overheated() -> None:
    # strictly rising -> MFI saturates at 100 -> above overbought -> short
    out = mfi_strategy(_frame(np.arange(1, 61, dtype=float)), period=14)
    assert out["signal"].iloc[-1] == -1


def test_allow_short_false_suppresses_shorts() -> None:
    out = mfi_strategy(_frame(np.arange(1, 61, dtype=float)), period=14, allow_short=False)
    assert (out["signal"] >= 0).all()


def test_does_not_mutate_input(sample_ohlcv: pd.DataFrame) -> None:
    before = set(sample_ohlcv.columns)
    mfi_strategy(sample_ohlcv, period=14)
    assert set(sample_ohlcv.columns) == before


def test_missing_columns_raise() -> None:
    with pytest.raises(ValueError, match="must contain columns"):
        mfi_strategy(pd.DataFrame({"close": [1.0, 2.0, 3.0]}))


def test_bad_period_raises() -> None:
    with pytest.raises(ValueError, match="period"):
        mfi_strategy(_frame(np.arange(1, 40, dtype=float)), period=0)


def test_unordered_levels_raise() -> None:
    with pytest.raises(ValueError, match="levels must satisfy"):
        mfi_strategy(_frame(np.arange(1, 40, dtype=float)), oversold=80, overbought=20)
