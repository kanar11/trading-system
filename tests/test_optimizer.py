"""Tests for the covariance-aware portfolio optimiser."""

import numpy as np
import pandas as pd
import pytest

from src.portfolio.optimizer import (
    min_variance_weights,
    max_sharpe_weights,
    risk_parity_weights,
)


def _synthetic_returns(n: int = 500, seed: int = 0) -> pd.DataFrame:
    """3-asset return panel with different vols + mild correlation."""
    rng = np.random.default_rng(seed)
    n_assets = 3
    cov = np.array(
        [[0.0004, 0.0001, 0.00005],
         [0.0001, 0.0009, 0.0002],
         [0.00005, 0.0002, 0.0016]],
    )
    means = np.array([0.0005, 0.0003, 0.0008])
    samples = rng.multivariate_normal(means, cov, size=n)
    return pd.DataFrame(samples, columns=["AAA", "BBB", "CCC"])


# ---------------------------------------------------------------------------
# min_variance_weights
# ---------------------------------------------------------------------------

def test_min_variance_weights_sum_to_one():
    rets = _synthetic_returns()
    w = min_variance_weights(rets)
    assert w.sum() == pytest.approx(1.0, abs=1e-9)
    assert (w >= 0).all()
    # series labels preserved
    assert list(w.index) == list(rets.columns)


def test_min_variance_overweights_low_vol_asset():
    rets = _synthetic_returns()
    w = min_variance_weights(rets)
    # AAA has the lowest variance — should attract the largest weight
    assert w["AAA"] == w.max()


def test_min_variance_accepts_precomputed_cov():
    rets = _synthetic_returns()
    cov = rets.cov()
    w1 = min_variance_weights(rets)
    w2 = min_variance_weights(rets, cov=cov)
    pd.testing.assert_series_equal(w1, w2, check_names=False)


# ---------------------------------------------------------------------------
# max_sharpe_weights
# ---------------------------------------------------------------------------

def test_max_sharpe_weights_sum_to_one():
    rets = _synthetic_returns()
    w = max_sharpe_weights(rets)
    assert w.sum() == pytest.approx(1.0, abs=1e-9)
    assert (w >= 0).all()


def test_max_sharpe_responds_to_rf_rate():
    rets = _synthetic_returns()
    w0 = max_sharpe_weights(rets, rf_daily=0.0)
    w_hi = max_sharpe_weights(rets, rf_daily=0.0005)
    # raising rf reduces effective excess returns, so the highest-return
    # asset gets relatively more weight (or stays max). At minimum the
    # vectors should differ.
    assert not np.allclose(w0.values, w_hi.values)


# ---------------------------------------------------------------------------
# risk_parity_weights
# ---------------------------------------------------------------------------

def test_risk_parity_weights_sum_to_one():
    rets = _synthetic_returns()
    w = risk_parity_weights(rets)
    assert w.sum() == pytest.approx(1.0, abs=1e-9)
    assert (w > 0).all()


def test_risk_parity_equalises_risk_contribution():
    rets = _synthetic_returns(n=1000, seed=1)
    w = risk_parity_weights(rets)
    cov = rets.cov().values
    marginal = cov @ w.values
    rc = w.values * marginal
    # all risk contributions should be approximately equal
    assert rc.std() / rc.mean() < 1e-3


def test_risk_parity_diagonal_cov_matches_inverse_vol():
    # with a diagonal covariance, risk-parity = 1 / sigma weighting
    rets = pd.DataFrame(
        np.random.RandomState(0).normal(0, 1, (500, 3)),
        columns=["x", "y", "z"],
    )
    sigmas = np.array([0.01, 0.02, 0.04])
    rets = rets * sigmas  # rescale to known vols, zero off-diagonal cov
    cov = np.diag(sigmas ** 2)
    w = risk_parity_weights(rets, cov=cov)
    expected = (1 / sigmas) / (1 / sigmas).sum()
    np.testing.assert_allclose(w.values, expected, rtol=1e-4)


# ---------------------------------------------------------------------------
# shared edge cases
# ---------------------------------------------------------------------------

def test_singular_cov_falls_back_to_equal_weights():
    # rank-deficient covariance — two identical columns
    n = 100
    base = pd.Series(np.random.RandomState(0).normal(0, 0.01, n))
    rets = pd.DataFrame({"A": base, "B": base, "C": base})
    w = min_variance_weights(rets)
    assert w.sum() == pytest.approx(1.0, abs=1e-6)
    # all weights should be equal (or very close, given the ridge)
    assert w.std() < 0.05


def test_bad_cov_shape_raises():
    rets = _synthetic_returns()
    bad_cov = np.eye(2)  # 2x2 but 3 assets
    with pytest.raises(ValueError, match="does not match"):
        min_variance_weights(rets, cov=bad_cov)


def test_bad_cov_non_square_raises():
    rets = _synthetic_returns()
    bad_cov = np.zeros((3, 2))  # not square at all
    with pytest.raises(ValueError, match="square"):
        min_variance_weights(rets, cov=bad_cov)
