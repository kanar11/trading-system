"""Tests for the additional risk metrics (skew/kurtosis, IR, Sterling, Burke)."""

import math

import numpy as np
import pandas as pd

from src.risk.metrics import (
    burke_ratio,
    information_ratio,
    kurtosis,
    skewness,
    sterling_ratio,
    tracking_error,
)

# --- skewness --------------------------------------------------------------


def test_skewness_symmetric_is_zero() -> None:
    assert abs(skewness(pd.Series([-2.0, -1.0, 0.0, 1.0, 2.0]))) < 1e-12


def test_skewness_sign() -> None:
    right = pd.Series([0.0] * 9 + [10.0])
    left = pd.Series([0.0] * 9 + [-10.0])
    assert skewness(right) > 0
    assert skewness(left) < 0


def test_skewness_guards() -> None:
    assert skewness(pd.Series([1.0, 2.0])) == 0.0
    assert skewness(pd.Series([3.0, 3.0, 3.0, 3.0])) == 0.0


# --- kurtosis --------------------------------------------------------------


def test_kurtosis_heavy_vs_flat_tails() -> None:
    leptokurtic = pd.Series([0.0] * 20 + [-5.0, 5.0])
    platykurtic = pd.Series([-2.0, -1.0, 0.0, 1.0, 2.0])
    assert kurtosis(leptokurtic) > 0
    assert kurtosis(platykurtic) < 0


def test_kurtosis_guards() -> None:
    assert kurtosis(pd.Series([1.0, 2.0, 3.0])) == 0.0
    assert kurtosis(pd.Series([3.0, 3.0, 3.0, 3.0])) == 0.0


# --- tracking error / information ratio ------------------------------------


def test_tracking_error_zero_when_identical() -> None:
    r = pd.Series([0.01, -0.02, 0.03, 0.0, 0.015])
    assert tracking_error(r, r) == 0.0


def test_tracking_error_positive_when_different() -> None:
    rng = np.random.default_rng(0)
    bench = pd.Series(rng.normal(0, 0.01, 300))
    strat = bench + pd.Series(rng.normal(0, 0.005, 300))
    assert tracking_error(strat, bench) > 0


def test_information_ratio_positive_on_outperformance() -> None:
    rng = np.random.default_rng(1)
    bench = pd.Series(rng.normal(0, 0.01, 400))
    strat = bench + pd.Series(rng.normal(0.001, 0.004, 400))
    assert information_ratio(strat, bench) > 0


def test_information_ratio_zero_when_identical() -> None:
    r = pd.Series([0.01, -0.02, 0.03, 0.0, 0.01, 0.02])
    assert information_ratio(r, r) == 0.0  # active is exactly 0 -> zero tracking error


# --- Sterling / Burke ------------------------------------------------------


def test_sterling_and_burke_finite_with_drawdown() -> None:
    r = pd.Series([0.1, 0.1, -0.15, 0.1, 0.1])
    s = sterling_ratio(r)
    b = burke_ratio(r)
    assert math.isfinite(s) and s > 0
    assert math.isfinite(b) and b > 0


def test_sterling_and_burke_infinite_without_drawdown() -> None:
    r = pd.Series([0.01] * 50)  # monotonic up -> no drawdown
    assert sterling_ratio(r) == math.inf
    assert burke_ratio(r) == math.inf


def test_drawdown_ratios_empty_is_zero() -> None:
    empty = pd.Series([], dtype=float)
    assert sterling_ratio(empty) == 0.0
    assert burke_ratio(empty) == 0.0
