"""Tests for the walk-forward validation module."""

import numpy as np
import pandas as pd
import pytest

from src.strategy.momentum import momentum_strategy
from src.backtest.engine import backtest_strategy
from src.validation.walk_forward import (
    WalkForwardConfig,
    run_walk_forward,
    print_walk_forward_report,
)


def _make_long_price_df(n_days: int = 800) -> pd.DataFrame:
    """Generate a synthetic price series for walk-forward testing."""
    np.random.seed(42)
    dates = pd.date_range("2015-01-01", periods=n_days, freq="B")
    returns = np.random.normal(0.0003, 0.01, n_days)
    prices = 100 * np.cumprod(1 + returns)
    return pd.DataFrame({"close": prices}, index=dates)


def _strategy_fn(df: pd.DataFrame) -> pd.DataFrame:
    return momentum_strategy(df, lookback=20, threshold=0.01)


def _backtest_fn(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    return backtest_strategy(df, transaction_cost=0.001)


class TestWalkForwardConfig:
    def test_default_step_equals_oos(self):
        cfg = WalkForwardConfig(in_sample_days=252, out_of_sample_days=63)
        assert cfg.step_days == 63

    def test_custom_step(self):
        cfg = WalkForwardConfig(in_sample_days=252, out_of_sample_days=63, step_days=21)
        assert cfg.step_days == 21


class TestRunWalkForward:
    def test_returns_expected_keys(self):
        df = _make_long_price_df(800)
        config = WalkForwardConfig(in_sample_days=300, out_of_sample_days=100)
        results = run_walk_forward(df, _strategy_fn, _backtest_fn, config)

        assert "folds" in results
        assert "summary" in results
        assert "oos_equity" in results
        assert "degradation" in results

    def test_produces_multiple_folds(self):
        df = _make_long_price_df(800)
        config = WalkForwardConfig(in_sample_days=200, out_of_sample_days=100)
        results = run_walk_forward(df, _strategy_fn, _backtest_fn, config)

        assert len(results["folds"]) >= 2

    def test_fold_has_correct_fields(self):
        df = _make_long_price_df(800)
        config = WalkForwardConfig(in_sample_days=300, out_of_sample_days=100)
        results = run_walk_forward(df, _strategy_fn, _backtest_fn, config)

        fold = results["folds"][0]
        assert fold.fold == 1
        assert fold.is_start is not None
        assert fold.oos_end is not None
        assert "Sharpe Ratio" in fold.oos_metrics
        assert "Sharpe Ratio" in fold.is_metrics

    def test_summary_has_avg_sharpe(self):
        df = _make_long_price_df(800)
        config = WalkForwardConfig(in_sample_days=300, out_of_sample_days=100)
        results = run_walk_forward(df, _strategy_fn, _backtest_fn, config)

        assert "Avg OOS Sharpe" in results["summary"]
        assert "Std OOS Sharpe" in results["summary"]
        assert "Total Folds" in results["summary"]

    def test_raises_on_insufficient_data(self):
        df = _make_long_price_df(100)
        config = WalkForwardConfig(in_sample_days=300, out_of_sample_days=100)

        with pytest.raises(ValueError, match="Not enough data"):
            run_walk_forward(df, _strategy_fn, _backtest_fn, config)

    def test_oos_equity_is_series(self):
        df = _make_long_price_df(800)
        config = WalkForwardConfig(in_sample_days=300, out_of_sample_days=100)
        results = run_walk_forward(df, _strategy_fn, _backtest_fn, config)

        assert isinstance(results["oos_equity"], pd.Series)
        assert len(results["oos_equity"]) > 0

    def test_degradation_keys(self):
        df = _make_long_price_df(800)
        config = WalkForwardConfig(in_sample_days=300, out_of_sample_days=100)
        results = run_walk_forward(df, _strategy_fn, _backtest_fn, config)

        d = results["degradation"]
        assert "avg_is_sharpe" in d
        assert "avg_oos_sharpe" in d
        assert "sharpe_degradation_pct" in d
