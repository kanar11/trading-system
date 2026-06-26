"""Tests for the factor-attribution module."""

import numpy as np
import pandas as pd
import pytest

from src.reporting.attribution import (
    AttributionResult,
    compute_beta,
    factor_regression,
)

# ---------------------------------------------------------------------------
# compute_beta
# ---------------------------------------------------------------------------


def test_beta_of_series_with_itself_is_one():
    rng = np.random.default_rng(0)
    bench = pd.Series(rng.normal(0, 0.01, 500))
    assert compute_beta(bench, bench) == pytest.approx(1.0, rel=1e-9)


def test_beta_of_constant_benchmark_is_zero():
    bench = pd.Series([0.001] * 100)
    strat = pd.Series(np.random.RandomState(0).normal(0, 0.01, 100))
    assert compute_beta(strat, bench) == 0.0


def test_beta_of_scaled_series():
    rng = np.random.default_rng(2)
    bench = pd.Series(rng.normal(0, 0.01, 500))
    strat = 2.5 * bench + pd.Series(rng.normal(0, 1e-9, 500))
    # negligible noise → beta ≈ 2.5
    assert compute_beta(strat, bench) == pytest.approx(2.5, rel=1e-3)


# ---------------------------------------------------------------------------
# factor_regression
# ---------------------------------------------------------------------------


def _synthetic(n: int = 1000, seed: int = 0):
    """Build factors + a strategy with known alpha and known betas."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    mkt = pd.Series(rng.normal(0.0004, 0.010, n), index=dates, name="MKT")
    smb = pd.Series(rng.normal(0.0001, 0.006, n), index=dates, name="SMB")
    hml = pd.Series(rng.normal(0.0002, 0.005, n), index=dates, name="HML")
    factors = pd.concat([mkt, smb, hml], axis=1)

    true_alpha = 0.0008  # daily alpha
    true_betas = {"MKT": 0.6, "SMB": -0.2, "HML": 0.4}
    noise = pd.Series(rng.normal(0, 0.004, n), index=dates)
    strategy = (
        true_alpha
        + true_betas["MKT"] * mkt
        + true_betas["SMB"] * smb
        + true_betas["HML"] * hml
        + noise
    )
    return strategy, factors, true_alpha, true_betas


def test_recovers_known_alpha_and_betas():
    strategy, factors, true_alpha, true_betas = _synthetic(n=2000, seed=42)
    res = factor_regression(strategy, factors)

    assert isinstance(res, AttributionResult)
    assert res.n_obs == 2000
    # tolerate ~10% relative error given finite-sample noise
    assert res.alpha_daily == pytest.approx(true_alpha, abs=2e-4)
    for name, beta in true_betas.items():
        assert res.betas[name] == pytest.approx(beta, abs=0.05)


def test_alpha_annualised_is_daily_times_252():
    strategy, factors, _, _ = _synthetic(n=500, seed=1)
    res = factor_regression(strategy, factors)
    assert res.alpha_annualised == pytest.approx(res.alpha_daily * 252, rel=1e-12)


def test_high_r_squared_when_fit_is_good():
    strategy, factors, _, _ = _synthetic(n=2000, seed=3)
    res = factor_regression(strategy, factors)
    # synthetic series with low noise — R² should be substantial
    assert res.r_squared > 0.5


def test_residuals_have_correct_length_and_zero_mean():
    strategy, factors, _, _ = _synthetic(n=1000, seed=4)
    res = factor_regression(strategy, factors)
    assert len(res.residuals) == 1000
    # OLS guarantees residuals sum to zero
    assert abs(res.residuals.sum()) < 1e-8


def test_raises_when_too_few_observations():
    dates = pd.date_range("2020-01-01", periods=3, freq="B")
    strat = pd.Series([0.01, 0.02, -0.01], index=dates)
    factors = pd.DataFrame({"A": [0.01, 0.0, -0.01], "B": [0.0, 0.01, 0.02]}, index=dates)
    with pytest.raises(ValueError, match="observations"):
        factor_regression(strat, factors)


def test_scalar_rf_subtraction():
    strategy, factors, true_alpha, _ = _synthetic(n=2000, seed=5)
    # subtracting a constant rf shifts alpha by -rf (per day)
    rf = 0.0002
    res = factor_regression(strategy, factors, rf_rate=rf)
    assert res.alpha_daily == pytest.approx(true_alpha - rf, abs=2e-4)
