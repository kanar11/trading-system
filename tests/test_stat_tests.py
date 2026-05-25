"""Tests for the statistical-significance test module."""

import math

import numpy as np
import pandas as pd
import pytest

from src.validation.stat_tests import (
    SharpeTestResult,
    sharpe_ttest,
    probabilistic_sharpe_ratio,
    deflated_sharpe_ratio,
    _norm_cdf,
    _norm_quantile,
)


# ---------------------------------------------------------------------------
# helpers: erf-based normal CDF & Acklam inverse
# ---------------------------------------------------------------------------

def test_norm_cdf_midpoint_is_half():
    assert _norm_cdf(0.0) == pytest.approx(0.5, abs=1e-12)


def test_norm_cdf_symmetric():
    for x in (0.1, 0.5, 1.0, 1.96, 3.0):
        assert _norm_cdf(x) + _norm_cdf(-x) == pytest.approx(1.0, abs=1e-12)


def test_norm_quantile_roundtrip():
    for p in (0.01, 0.1, 0.5, 0.9, 0.99):
        z = _norm_quantile(p)
        assert _norm_cdf(z) == pytest.approx(p, abs=1e-6)


def test_norm_quantile_out_of_range_raises():
    with pytest.raises(ValueError):
        _norm_quantile(0.0)
    with pytest.raises(ValueError):
        _norm_quantile(1.0)


# ---------------------------------------------------------------------------
# sharpe_ttest
# ---------------------------------------------------------------------------

def test_sharpe_ttest_zero_returns():
    res = sharpe_ttest(pd.Series(np.zeros(100)))
    assert isinstance(res, SharpeTestResult)
    assert res.sharpe_annualised == 0.0
    assert res.p_value_two_sided == pytest.approx(1.0, abs=1e-9)


def test_sharpe_ttest_clearly_positive():
    # mean 0.001 / std 0.005 → SR_daily=0.2 → SR_ann ≈ 3.18, t huge
    np.random.seed(0)
    r = pd.Series(np.random.normal(0.001, 0.005, 1000))
    res = sharpe_ttest(r)
    assert res.sharpe_annualised > 1.0
    assert res.p_value_two_sided < 0.01


def test_sharpe_ttest_too_few_observations():
    res = sharpe_ttest(pd.Series([0.01]))
    assert res.n_obs == 1
    assert res.p_value_two_sided == 1.0


# ---------------------------------------------------------------------------
# probabilistic_sharpe_ratio
# ---------------------------------------------------------------------------

def test_psr_at_target_equal_to_realised_is_half():
    np.random.seed(1)
    r = pd.Series(np.random.normal(0.001, 0.01, 500))
    # Use the realised annualised Sharpe as the target → PSR should be ~0.5
    sr_ann = sharpe_ttest(r).sharpe_annualised
    psr = probabilistic_sharpe_ratio(r, target_sharpe=sr_ann)
    assert psr == pytest.approx(0.5, abs=0.05)


def test_psr_target_above_realised_below_half():
    np.random.seed(2)
    r = pd.Series(np.random.normal(0.0005, 0.01, 500))
    sr_ann = sharpe_ttest(r).sharpe_annualised
    psr_high = probabilistic_sharpe_ratio(r, target_sharpe=sr_ann + 1.0)
    psr_low = probabilistic_sharpe_ratio(r, target_sharpe=sr_ann - 1.0)
    assert psr_high < 0.5 < psr_low


def test_psr_returns_in_unit_interval():
    np.random.seed(3)
    r = pd.Series(np.random.normal(0.001, 0.01, 200))
    psr = probabilistic_sharpe_ratio(r, target_sharpe=0.5)
    assert 0.0 <= psr <= 1.0


# ---------------------------------------------------------------------------
# deflated_sharpe_ratio
# ---------------------------------------------------------------------------

def test_dsr_with_one_trial_matches_psr_against_zero():
    np.random.seed(4)
    r = pd.Series(np.random.normal(0.001, 0.01, 500))
    dsr = deflated_sharpe_ratio(r, n_trials=1)
    psr = probabilistic_sharpe_ratio(r, target_sharpe=0.0)
    assert dsr == pytest.approx(psr, abs=1e-9)


def test_dsr_decreases_with_more_trials():
    np.random.seed(5)
    r = pd.Series(np.random.normal(0.001, 0.01, 500))
    dsr_1 = deflated_sharpe_ratio(r, n_trials=1)
    dsr_100 = deflated_sharpe_ratio(r, n_trials=100)
    dsr_1000 = deflated_sharpe_ratio(r, n_trials=1000)
    assert dsr_1 > dsr_100 > dsr_1000


def test_dsr_returns_in_unit_interval():
    np.random.seed(6)
    r = pd.Series(np.random.normal(0.0005, 0.01, 200))
    for trials in (1, 10, 100, 10_000):
        d = deflated_sharpe_ratio(r, n_trials=trials)
        assert 0.0 <= d <= 1.0
