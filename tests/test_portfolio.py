"""Tests for the multi-asset portfolio backtest."""

import numpy as np
import pandas as pd
import pytest

from src.backtest.engine import backtest_strategy
from src.strategy.momentum import momentum_strategy
from src.portfolio.portfolio import (
    PortfolioConfig,
    PortfolioResult,
    run_portfolio_backtest,
)


def _make_synthetic(seed: int, n: int = 200) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    rets = rng.normal(0.0005, 0.012, n)
    close = 100 * np.exp(np.cumsum(rets))
    return pd.DataFrame(
        {
            "open": close,
            "high": close * 1.005,
            "low": close * 0.995,
            "close": close,
            "volume": rng.integers(1_000_000, 10_000_000, n),
        },
        index=dates,
    )


def _basket() -> dict[str, pd.DataFrame]:
    return {
        "AAA": _make_synthetic(seed=1),
        "BBB": _make_synthetic(seed=2),
        "CCC": _make_synthetic(seed=3),
    }


def _mom(df):
    return momentum_strategy(df, lookback=10, threshold=0.0, use_sma_filter=False)


def _bt(df):
    return backtest_strategy(df, transaction_cost=0.0, vol_target=None)


def test_equal_weight_runs_and_normalises():
    res = run_portfolio_backtest(_basket(), _mom, _bt, PortfolioConfig(weighting="equal"))
    assert isinstance(res, PortfolioResult)
    assert res.weights.shape[1] == 3
    # equal weights row-sum to 1
    assert np.allclose(res.weights.sum(axis=1).values, 1.0, atol=1e-9)


def test_inverse_vol_weights_sum_to_one():
    res = run_portfolio_backtest(
        _basket(), _mom, _bt,
        PortfolioConfig(weighting="inverse_vol", vol_window=20),
    )
    # after warm-up, each row should sum to 1
    row_sums = res.weights.iloc[30:].sum(axis=1).values
    assert np.allclose(row_sums, 1.0, atol=1e-9)


def test_custom_weights_applied_and_normalised():
    res = run_portfolio_backtest(
        _basket(), _mom, _bt,
        PortfolioConfig(weighting="custom", custom_weights={"AAA": 1, "BBB": 1, "CCC": 2}),
    )
    last_row = res.weights.iloc[-1]
    assert last_row["AAA"] == pytest.approx(0.25)
    assert last_row["BBB"] == pytest.approx(0.25)
    assert last_row["CCC"] == pytest.approx(0.50)


def test_custom_weights_required():
    with pytest.raises(ValueError, match="custom_weights"):
        run_portfolio_backtest(
            _basket(), _mom, _bt, PortfolioConfig(weighting="custom"),
        )


def test_unknown_scheme_raises():
    with pytest.raises(ValueError, match="unknown weighting"):
        run_portfolio_backtest(
            _basket(), _mom, _bt, PortfolioConfig(weighting="bogus"),
        )


def test_empty_data_raises():
    with pytest.raises(ValueError, match="at least one"):
        run_portfolio_backtest({}, _mom, _bt)


def test_portfolio_returns_match_weighted_sum():
    res = run_portfolio_backtest(_basket(), _mom, _bt, PortfolioConfig(weighting="equal"))
    manual = (res.returns * res.weights).sum(axis=1)
    pd.testing.assert_series_equal(
        res.portfolio_returns.reset_index(drop=True),
        manual.reset_index(drop=True),
        check_names=False,
    )


def test_per_asset_metrics_populated():
    res = run_portfolio_backtest(_basket(), _mom, _bt, PortfolioConfig(weighting="equal"))
    assert set(res.per_asset_metrics.keys()) == {"AAA", "BBB", "CCC"}
    for stats in res.per_asset_metrics.values():
        assert "Sharpe Ratio" in stats
