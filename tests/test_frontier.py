"""Tests for the mean-variance efficient frontier."""

import numpy as np
import pandas as pd
import pytest

from src.portfolio import efficient_frontier


def _returns(n: int = 400, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    data = {
        "low": rng.normal(0.0002, 0.005, n),
        "mid": rng.normal(0.0005, 0.010, n),
        "high": rng.normal(0.0010, 0.020, n),
    }
    return pd.DataFrame(data, index=idx)


def test_shapes_and_weight_sums() -> None:
    frontier = efficient_frontier(_returns(), n_points=15)
    assert len(frontier.expected_returns) == 15
    assert len(frontier.volatilities) == 15
    assert len(frontier.sharpe_ratios) == 15
    assert frontier.weights.shape == (15, 3)
    assert list(frontier.weights.columns) == ["low", "mid", "high"]
    assert np.allclose(frontier.weights.sum(axis=1).to_numpy(), 1.0)


def test_long_only_by_default() -> None:
    frontier = efficient_frontier(_returns())
    assert (frontier.weights.to_numpy() >= 0).all()


def test_allow_short_hits_targets_exactly() -> None:
    df = _returns()
    frontier = efficient_frontier(df, n_points=10, allow_short=True)
    mu = df.mean().to_numpy()
    targets = np.linspace(mu.min(), mu.max(), 10)
    assert np.allclose(frontier.expected_returns, targets)


def test_min_volatility_point_is_global_minimum() -> None:
    frontier = efficient_frontier(_returns(), n_points=25, allow_short=True)
    idx = frontier.min_volatility_index
    assert frontier.volatilities[idx] == frontier.volatilities.min()
    # interior points near the min-var portfolio are less volatile than the
    # extreme single-asset-mean endpoints
    assert frontier.volatilities[idx] < frontier.volatilities[-1]


def test_expected_returns_increase_along_frontier() -> None:
    frontier = efficient_frontier(_returns(), n_points=10, allow_short=True)
    assert (np.diff(frontier.expected_returns) >= -1e-12).all()


def test_max_sharpe_index_is_argmax() -> None:
    frontier = efficient_frontier(_returns(), n_points=25)
    idx = frontier.max_sharpe_index
    assert frontier.sharpe_ratios[idx] == frontier.sharpe_ratios.max()


def test_single_asset_frontier_is_all_in() -> None:
    df = _returns()[["mid"]]
    frontier = efficient_frontier(df, n_points=5)
    assert np.allclose(frontier.weights.to_numpy(), 1.0)


def test_equal_means_collapse_to_min_variance() -> None:
    # both columns have mean exactly 0 -> degenerate frontier
    df = pd.DataFrame(
        {
            "a": [0.01, -0.01, 0.02, -0.02, 0.015, -0.015],
            "b": [-0.02, 0.02, 0.01, -0.01, -0.005, 0.005],
        }
    )
    frontier = efficient_frontier(df, n_points=5)
    assert np.isfinite(frontier.weights.to_numpy()).all()
    first_row = frontier.weights.iloc[0].to_numpy()
    for i in range(1, 5):
        assert np.allclose(frontier.weights.iloc[i].to_numpy(), first_row)


def test_accepts_explicit_covariance() -> None:
    df = _returns()
    frontier = efficient_frontier(df, n_points=8, cov=df.cov())
    assert len(frontier.expected_returns) == 8
    assert np.isfinite(frontier.volatilities).all()


def test_too_few_points_raises() -> None:
    with pytest.raises(ValueError, match="n_points"):
        efficient_frontier(_returns(), n_points=1)
