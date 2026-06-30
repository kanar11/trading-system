"""Tests for CAPM / benchmark-relative metrics (Treynor, Jensen, M2)."""

import pandas as pd
import pytest

from src.risk.metrics import jensen_alpha, m2_ratio, treynor_ratio

_IDX = pd.date_range("2020-01-01", periods=6, freq="B")
_BENCH = pd.Series([0.01, -0.005, 0.02, 0.0, 0.015, -0.01], index=_IDX)
_MEAN = _BENCH.mean()  # 0.005


def test_treynor_self_is_annualised_mean() -> None:
    # strategy == benchmark -> beta 1 -> Treynor = annualised mean excess
    assert treynor_ratio(_BENCH, _BENCH) == pytest.approx(_MEAN * 252)


def test_jensen_alpha_zero_for_self() -> None:
    assert jensen_alpha(_BENCH, _BENCH) == pytest.approx(0.0, abs=1e-12)


def test_jensen_alpha_constant_outperformance() -> None:
    strat = _BENCH + 0.001  # beta stays 1, pure alpha of 0.001/day
    assert jensen_alpha(strat, _BENCH) == pytest.approx(0.001 * 252)


def test_m2_self_is_annualised_mean() -> None:
    assert m2_ratio(_BENCH, _BENCH) == pytest.approx(_MEAN * 252)


def test_zero_variance_benchmark_returns_zero() -> None:
    flat = pd.Series([0.01] * 6, index=_IDX)
    assert treynor_ratio(_BENCH, flat) == 0.0
    assert jensen_alpha(_BENCH, flat) == 0.0
    assert m2_ratio(_BENCH, flat) == 0.0


def test_zero_volatility_strategy_m2_zero() -> None:
    flat = pd.Series([0.0] * 6, index=_IDX)
    assert m2_ratio(flat, _BENCH) == 0.0
    assert treynor_ratio(flat, _BENCH) == 0.0  # beta 0


def test_insufficient_overlap_returns_zero() -> None:
    a = pd.Series([0.01], index=_IDX[:1])
    assert treynor_ratio(a, a) == 0.0
    assert jensen_alpha(a, a) == 0.0
    assert m2_ratio(a, a) == 0.0
