"""Tests for Stochastic RSI and the Choppiness Index."""

import numpy as np
import pandas as pd
import pytest

from src.indicators import choppiness, stoch_rsi


def _trend(n: int = 120, step: float = 1.0) -> tuple[pd.Series, pd.Series, pd.Series]:
    base = 100.0 + step * np.arange(n, dtype=float)
    return pd.Series(base + 0.5), pd.Series(base - 0.5), pd.Series(base)


def _sideways(n: int = 120) -> tuple[pd.Series, pd.Series, pd.Series]:
    close = pd.Series(100.0 + 3.0 * np.sin(np.arange(n, dtype=float)))
    return close + 0.5, close - 0.5, close


# --- Stochastic RSI ---------------------------------------------------------


def test_stoch_rsi_columns_and_bounds() -> None:
    _, _, close = _sideways()
    out = stoch_rsi(close)
    assert list(out.columns) == ["stoch_rsi", "k", "d"]
    valid = out.dropna()
    assert len(valid) > 0
    assert ((valid >= 0) & (valid <= 100)).all().all()


def test_pins_to_extremes_in_a_persistent_trend() -> None:
    _, _, close = _trend(step=1.0)
    out = stoch_rsi(close)
    # RSI keeps making new highs -> stochastic of RSI sits at 100
    assert (out["stoch_rsi"].dropna().iloc[-20:] == pytest.approx(100.0)).all()
    _, _, falling = _trend(step=-1.0)
    down = stoch_rsi(falling)
    assert (down["stoch_rsi"].dropna().iloc[-20:] == pytest.approx(0.0)).all()


def test_k_and_d_are_smoothed_versions() -> None:
    _, _, close = _sideways()
    out = stoch_rsi(close).dropna()
    # smoothing shrinks bar-to-bar variation
    assert out["k"].diff().abs().mean() <= out["stoch_rsi"].diff().abs().mean()
    assert out["d"].diff().abs().mean() <= out["k"].diff().abs().mean()


def test_stoch_rsi_bad_params_raise() -> None:
    _, _, close = _sideways()
    with pytest.raises(ValueError, match=">= 1"):
        stoch_rsi(close, rsi_period=0)
    with pytest.raises(ValueError, match=">= 1"):
        stoch_rsi(close, smooth_k=0)


# --- Choppiness Index -------------------------------------------------------


def test_straight_trend_scores_near_zero() -> None:
    high, low, close = _trend(step=2.0)
    out = choppiness(high, low, close).dropna()
    assert (out < 38.2).all()  # classic trending threshold


def test_sideways_market_scores_high() -> None:
    high, low, close = _sideways()
    trend_out = choppiness(*_trend(step=2.0)).dropna()
    chop_out = choppiness(high, low, close).dropna()
    assert chop_out.mean() > trend_out.mean() + 20
    assert chop_out.max() <= 100.0


def test_choppiness_warm_up_and_bounds() -> None:
    high, low, close = _sideways()
    out = choppiness(high, low, close, period=14)
    assert out.iloc[:13].isna().all()
    valid = out.dropna()
    assert ((valid >= 0) & (valid <= 100)).all()


def test_choppiness_bad_period_raises() -> None:
    high, low, close = _sideways()
    with pytest.raises(ValueError, match="period"):
        choppiness(high, low, close, period=1)
