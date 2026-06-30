"""Tests for up/down market capture ratios."""

import math

import pandas as pd
import pytest

from src.reporting.attribution import capture_ratio, down_capture, up_capture

_IDX = pd.date_range("2020-01-01", periods=4, freq="B")
_BENCH = pd.Series([0.02, -0.01, 0.03, -0.02], index=_IDX)


def test_identical_series_capture_is_one() -> None:
    assert up_capture(_BENCH, _BENCH) == pytest.approx(1.0)
    assert down_capture(_BENCH, _BENCH) == pytest.approx(1.0)
    assert capture_ratio(_BENCH, _BENCH) == pytest.approx(1.0)


def test_half_beta_captures_half_each_side() -> None:
    strat = _BENCH * 0.5
    assert up_capture(strat, _BENCH) == pytest.approx(0.5)
    assert down_capture(strat, _BENCH) == pytest.approx(0.5)
    assert capture_ratio(strat, _BENCH) == pytest.approx(1.0)


def test_amplified_up_dampened_down() -> None:
    strat = pd.Series([0.03, -0.005, 0.045, -0.01], index=_IDX)  # up x1.5, down x0.5
    assert up_capture(strat, _BENCH) == pytest.approx(1.5)
    assert down_capture(strat, _BENCH) == pytest.approx(0.5)
    assert capture_ratio(strat, _BENCH) == pytest.approx(3.0)


def test_defensive_zero_down_capture_gives_inf_ratio() -> None:
    strat = pd.Series([0.02, 0.0, 0.03, 0.0], index=_IDX)  # flat on down days
    assert down_capture(strat, _BENCH) == pytest.approx(0.0)
    assert capture_ratio(strat, _BENCH) == math.inf


def test_no_up_days_returns_zero_up_capture() -> None:
    idx = pd.date_range("2020-01-01", periods=2, freq="B")
    bench = pd.Series([-0.01, -0.02], index=idx)
    assert up_capture(bench, bench) == 0.0
    assert down_capture(bench, bench) == pytest.approx(1.0)


def test_no_overlap_returns_zero() -> None:
    a = pd.Series([0.01, 0.02], index=pd.to_datetime(["2020-01-01", "2020-01-02"]))
    b = pd.Series([0.01, 0.02], index=pd.to_datetime(["2021-01-01", "2021-01-02"]))
    assert up_capture(a, b) == 0.0
    assert down_capture(a, b) == 0.0
