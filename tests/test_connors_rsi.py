"""Tests for the Connors RSI composite oscillator."""

import numpy as np
import pandas as pd
import pytest

from src.indicators import connors_rsi


def _random_walk(n: int = 300, seed: int = 1) -> pd.Series:
    rng = np.random.default_rng(seed)
    return pd.Series(100.0 + np.cumsum(rng.normal(0.0, 1.0, n)))


def test_bounded_in_zero_hundred() -> None:
    crsi = connors_rsi(_random_walk(), rank_lookback=50).dropna()
    assert crsi.name == "connors_rsi"
    assert (crsi >= 0.0).all()
    assert (crsi <= 100.0).all()


def test_pure_downtrend_floors_at_zero() -> None:
    close = pd.Series(200.0 - np.arange(160, dtype=float))
    crsi = connors_rsi(close, rank_lookback=50).dropna()
    # price RSI 0, streak RSI 0, and shrinking negative returns rank last -> 0
    assert crsi.iloc[-1] == pytest.approx(0.0)


def test_linear_uptrend_maxes_the_two_rsi_components() -> None:
    # a linear (arithmetic) up-trend maxes both RSI legs (100) but the
    # per-bar return shrinks each step, so the percent-rank leg floors at 0
    close = pd.Series(100.0 + np.arange(160, dtype=float))
    crsi = connors_rsi(close, rank_lookback=50).dropna()
    assert np.allclose(crsi.to_numpy(), 200.0 / 3.0)


def test_reacts_to_sharp_return_spikes() -> None:
    rng = np.random.default_rng(3)
    r = rng.normal(0.0, 0.005, 120)
    r[110] = 0.06  # a big up move
    r[115] = -0.06  # a big down move
    close = pd.Series(100.0 * np.cumprod(1 + r))
    crsi = connors_rsi(close, rank_lookback=50)
    median = crsi.dropna().median()
    assert crsi.iloc[110] > median + 30  # overbought spike
    assert crsi.iloc[115] < median - 30  # oversold crash


def test_warm_up_covers_the_rank_window() -> None:
    close = _random_walk()
    crsi = connors_rsi(close, rank_lookback=50)
    # the percent-rank leg needs the longest warm-up (rank_lookback bars)
    assert crsi.iloc[:50].isna().all()
    assert crsi.iloc[60:].notna().all()


def test_shorter_rank_lookback_shortens_warm_up() -> None:
    close = _random_walk()
    long_wu = connors_rsi(close, rank_lookback=100).notna().argmax()
    short_wu = connors_rsi(close, rank_lookback=20).notna().argmax()
    assert short_wu < long_wu


def test_bad_params_raise() -> None:
    close = _random_walk(60)
    with pytest.raises(ValueError, match="must be >= 2"):
        connors_rsi(close, rsi_period=1)
    with pytest.raises(ValueError, match="must be >= 2"):
        connors_rsi(close, streak_period=1)
    with pytest.raises(ValueError, match="rank_lookback"):
        connors_rsi(close, rank_lookback=1)
