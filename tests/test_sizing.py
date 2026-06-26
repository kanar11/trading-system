"""Tests for the position-sizing helpers."""

import numpy as np
import pandas as pd
import pytest

from src.risk.sizing import (
    atr_position_size,
    fixed_fractional,
    kelly_fraction,
)

# ---------------------------------------------------------------------------
# kelly_fraction
# ---------------------------------------------------------------------------


def test_kelly_zero_on_empty_input():
    assert kelly_fraction(pd.Series(dtype=float)) == 0.0


def test_kelly_zero_on_negative_edge():
    losses = pd.Series([-0.01, -0.02, -0.005, -0.015])
    assert kelly_fraction(losses) == 0.0


def test_kelly_positive_on_positive_edge():
    np.random.seed(0)
    # mean +0.002, vol 0.01 — clearly positive Kelly
    returns = pd.Series(np.random.normal(0.002, 0.01, 1000))
    f = kelly_fraction(returns, cap=1.0)
    assert 0.0 < f <= 1.0


def test_kelly_respects_cap():
    # huge positive edge so unclipped Kelly > 1
    returns = pd.Series([0.5, 0.4, 0.6, 0.55, 0.5])
    assert kelly_fraction(returns, cap=0.25) == 0.25


def test_kelly_half_is_half_of_full():
    np.random.seed(1)
    returns = pd.Series(np.random.normal(0.001, 0.01, 5000))
    # cap must be loose enough that neither call clips, otherwise
    # the "half == full/2" identity breaks down
    full = kelly_fraction(returns, cap=1000.0, kelly_fraction_of_full=1.0)
    half = kelly_fraction(returns, cap=1000.0, kelly_fraction_of_full=0.5)
    assert full > 0
    assert half == pytest.approx(full / 2, rel=1e-9)


# ---------------------------------------------------------------------------
# atr_position_size
# ---------------------------------------------------------------------------


def test_atr_size_zero_on_bad_inputs():
    assert atr_position_size(0, 1.0, 100_000) == 0.0
    assert atr_position_size(100, 0, 100_000) == 0.0
    assert atr_position_size(100, 1.0, 0) == 0.0


def test_atr_size_respects_risk_budget():
    # risk 1% of 100k on a 2*ATR stop with ATR=1 and price=100
    # dollar risk = 1000, units = 1000 / 2 = 500, notional = 500 * 100 = 50_000
    # fraction = 50_000 / 100_000 = 0.5
    f = atr_position_size(
        price=100,
        atr=1.0,
        equity=100_000,
        risk_per_trade=0.01,
        atr_multiple=2.0,
        max_size=10.0,
    )
    assert f == pytest.approx(0.5)


def test_atr_size_capped():
    f = atr_position_size(
        price=100,
        atr=0.5,
        equity=100_000,
        risk_per_trade=0.05,
        atr_multiple=2.0,
        max_size=1.0,
    )
    assert f == 1.0


# ---------------------------------------------------------------------------
# fixed_fractional
# ---------------------------------------------------------------------------


def test_fixed_fractional_zero_on_no_edge():
    # win-rate 50% with 1:1 payoff is breakeven
    assert fixed_fractional(0.5, 1.0) == 0.0


def test_fixed_fractional_zero_on_bad_input():
    assert fixed_fractional(0.0, 2.0) == 0.0
    assert fixed_fractional(1.0, 2.0) == 0.0
    assert fixed_fractional(0.5, 0.0) == 0.0


def test_fixed_fractional_classic_example():
    # 60% wins with 2:1 payoff → 0.6 - 0.4/2 = 0.4
    assert fixed_fractional(0.6, 2.0) == pytest.approx(0.4)


def test_fixed_fractional_respects_cap():
    assert fixed_fractional(0.9, 5.0, cap=0.25) == 0.25
