"""Tests for the public ADX / DI indicator."""

import numpy as np
import pandas as pd
import pytest

from src.indicators import adx
from src.regime.detector import _adx


def _trend_bars(n: int = 120, step: float = 1.0) -> tuple[pd.Series, pd.Series, pd.Series]:
    base = 100.0 + step * np.arange(n, dtype=float)
    return pd.Series(base + 0.6), pd.Series(base - 0.6), pd.Series(base)


def _chop_bars(n: int = 200, seed: int = 5) -> tuple[pd.Series, pd.Series, pd.Series]:
    rng = np.random.default_rng(seed)
    close = pd.Series(100.0 + rng.normal(0, 0.4, n).cumsum() * 0.0 + rng.normal(0, 0.5, n))
    high = close + np.abs(rng.normal(0.5, 0.1, n))
    low = close - np.abs(rng.normal(0.5, 0.1, n))
    return high, low, close


def test_output_columns_and_bounds() -> None:
    high, low, close = _trend_bars()
    out = adx(high, low, close)
    assert list(out.columns) == ["adx", "di_plus", "di_minus"]
    valid = out.dropna()
    assert ((valid >= 0) & (valid <= 100)).all().all()


def test_matches_the_regime_detector_implementation() -> None:
    high, low, close = _trend_bars()
    ours = adx(high, low, close, period=14)["adx"]
    theirs = _adx(high, low, close, period=14)
    assert np.allclose(ours.dropna().to_numpy(), theirs.dropna().to_numpy())


def test_uptrend_has_di_plus_dominant_and_high_adx() -> None:
    high, low, close = _trend_bars(step=1.0)
    out = adx(high, low, close).dropna()
    assert (out["di_plus"].iloc[-20:] > out["di_minus"].iloc[-20:]).all()
    assert out["adx"].iloc[-1] > 25


def test_downtrend_flips_the_directional_lines() -> None:
    high, low, close = _trend_bars(step=-1.0)
    out = adx(high, low, close).dropna()
    assert (out["di_minus"].iloc[-20:] > out["di_plus"].iloc[-20:]).all()


def test_chop_scores_lower_than_trend() -> None:
    trend_out = adx(*_trend_bars())["adx"].dropna()
    chop_out = adx(*_chop_bars())["adx"].dropna()
    assert chop_out.iloc[-1] < trend_out.iloc[-1]


def test_warm_up_is_nan() -> None:
    high, low, close = _trend_bars()
    out = adx(high, low, close, period=14)
    assert out["adx"].iloc[:14].isna().all()


def test_bad_period_raises() -> None:
    high, low, close = _trend_bars(20)
    with pytest.raises(ValueError, match="period"):
        adx(high, low, close, period=0)
