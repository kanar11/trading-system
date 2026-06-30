"""Tests for backtest position-series exposure analytics."""

import pandas as pd
import pytest

from src.backtest.exposure import summarize_exposure


def test_mixed_position_path() -> None:
    # flat, long, long, short, flat
    s = summarize_exposure(pd.Series([0.0, 1.0, 1.0, -1.0, 0.0]))
    assert s.time_in_market == pytest.approx(0.6)
    assert s.avg_exposure == pytest.approx(0.6)
    assert s.long_fraction == pytest.approx(0.4)
    assert s.short_fraction == pytest.approx(0.2)
    assert s.avg_long_exposure == pytest.approx(1.0)
    assert s.avg_short_exposure == pytest.approx(1.0)
    # deltas from prior(0): [+1, 0, -2, +1] over bars 1..4 -> turnover 4, 3 changes
    assert s.turnover == pytest.approx(4.0)
    assert s.n_trades == 3


def test_constant_long_has_single_entry() -> None:
    s = summarize_exposure(pd.Series([1.0, 1.0, 1.0]))
    assert s.time_in_market == pytest.approx(1.0)
    assert s.long_fraction == pytest.approx(1.0)
    assert s.turnover == pytest.approx(1.0)  # only the initial entry from 0
    assert s.n_trades == 1


def test_fractional_exposure() -> None:
    s = summarize_exposure(pd.Series([0.0, 0.5, 0.5, 0.0]))
    assert s.avg_exposure == pytest.approx(0.25)
    assert s.avg_long_exposure == pytest.approx(0.5)
    assert s.turnover == pytest.approx(1.0)  # +0.5 then -0.5
    assert s.n_trades == 2


def test_all_flat_is_zero() -> None:
    s = summarize_exposure(pd.Series([0.0, 0.0, 0.0]))
    assert s.time_in_market == 0.0
    assert s.turnover == 0.0
    assert s.n_trades == 0
    assert s.avg_long_exposure == 0.0
    assert s.avg_short_exposure == 0.0


def test_empty_series_is_zero() -> None:
    s = summarize_exposure(pd.Series([], dtype=float))
    assert s.time_in_market == 0.0
    assert s.n_trades == 0
