"""Tests for the Monte-Carlo robustness module."""

import numpy as np
import pandas as pd
import pytest

from src.validation.monte_carlo import (
    MonteCarloResult,
    bootstrap_returns,
    shuffle_trade_log,
)


def test_bootstrap_returns_basic_shape():
    np.random.seed(0)
    returns = pd.Series(np.random.normal(0.001, 0.01, 200))
    res = bootstrap_returns(returns, n_simulations=50, seed=123)

    assert isinstance(res, MonteCarloResult)
    assert res.n_simulations == 50
    assert len(res.metric_samples) == 50
    # summary must have the same metrics as the per-simulation rows
    assert set(res.summary.index) == set(res.metric_samples.columns)
    assert {"mean", "std", "p05", "p50", "p95"} == set(res.summary.columns)


def test_bootstrap_is_reproducible_with_seed():
    returns = pd.Series(np.random.RandomState(0).normal(0.001, 0.01, 100))
    a = bootstrap_returns(returns, n_simulations=20, seed=7)
    b = bootstrap_returns(returns, n_simulations=20, seed=7)
    pd.testing.assert_frame_equal(a.metric_samples, b.metric_samples)


def test_bootstrap_block_mode_runs():
    returns = pd.Series(np.random.RandomState(1).normal(0, 0.01, 200))
    res = bootstrap_returns(returns, n_simulations=10, block_size=10, seed=1)
    assert len(res.metric_samples) == 10


def test_bootstrap_rejects_empty():
    with pytest.raises(ValueError):
        bootstrap_returns(pd.Series(dtype=float))


def test_bootstrap_rejects_bad_block():
    with pytest.raises(ValueError):
        bootstrap_returns(pd.Series([0.01, 0.02]), block_size=0)


def test_shuffle_trade_log_preserves_total_return():
    # arithmetic mean is invariant under permutation, but multiplicative
    # equity is NOT — however, the *set* of trades is identical, so the
    # final equity (1+r1)(1+r2)... is invariant under reordering.
    trades = pd.Series([0.02, -0.01, 0.03, -0.005, 0.01])
    original_total = float((1 + trades).prod() - 1)

    res = shuffle_trade_log(trades, n_simulations=50, seed=0)
    sampled_totals = res.metric_samples["Total Return"].round(10).unique()

    # all permutations should give the same total return
    assert len(sampled_totals) == 1
    assert sampled_totals[0] == pytest.approx(original_total, abs=1e-9)


def test_shuffle_max_dd_varies():
    # max drawdown IS path-dependent, so different orderings give
    # different values — but not always strictly different on tiny
    # series, so use a long enough one
    np.random.seed(2)
    trades = pd.Series(np.random.normal(0.001, 0.02, 200))
    res = shuffle_trade_log(trades, n_simulations=100, seed=0)

    unique_dds = res.metric_samples["Max Drawdown"].round(6).unique()
    assert len(unique_dds) > 1


def test_shuffle_rejects_empty():
    with pytest.raises(ValueError):
        shuffle_trade_log(pd.Series(dtype=float))
