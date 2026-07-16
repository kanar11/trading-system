"""Tests for weight-frame walk-forward evaluation."""

import numpy as np
import pandas as pd
import pytest

from src.backtest import backtest_weights, walk_forward_weights
from src.portfolio import min_variance_weights


def _prices(n: int = 200, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2023-01-02", periods=n, freq="B")
    data = {
        "aaa": 100.0 * np.cumprod(1 + rng.normal(0.0004, 0.01, n)),
        "bbb": 50.0 * np.cumprod(1 + rng.normal(0.0002, 0.008, n)),
    }
    return pd.DataFrame(data, index=idx)


def _equal_weights(train: pd.DataFrame) -> pd.Series:
    return pd.Series(1.0 / train.shape[1], index=train.columns)


def test_equal_weight_rule_matches_direct_backtest() -> None:
    prices = _prices()
    out = walk_forward_weights(prices, _equal_weights, train_window=60, test_window=20)
    manual = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    manual.iloc[59:] = 0.5  # first decision on the close of bar 59
    expected = backtest_weights(prices, manual, cost_bps=10.0)
    assert np.allclose(out.results["equity_curve"].to_numpy(), expected["equity_curve"].to_numpy())


def test_no_exposure_before_the_first_fold() -> None:
    out = walk_forward_weights(_prices(), _equal_weights, train_window=60, test_window=20)
    assert (out.weights.iloc[:59] == 0.0).all().all()
    assert (out.results["portfolio_return_gross"].iloc[:60] == 0.0).all()


def test_fold_count_covers_the_sample() -> None:
    out = walk_forward_weights(_prices(200), _equal_weights, train_window=60, test_window=25)
    # folds start at bars 60, 85, 110, 135, 160, 185
    assert out.n_folds == 6


def test_weight_fn_sees_only_history() -> None:
    prices = _prices()
    seen_ends: list[pd.Timestamp] = []

    def spy(train: pd.DataFrame) -> pd.Series:
        assert len(train) == 60
        seen_ends.append(pd.Timestamp(train.index[-1]))
        return _equal_weights(train)

    out = walk_forward_weights(prices, spy, train_window=60, test_window=20)
    # each fold's training data ends exactly on its decision bar
    decision_positions = list(range(59, len(prices) - 1, 20))
    expected_ends = [pd.Timestamp(prices.index[p]) for p in decision_positions]
    assert seen_ends[: len(expected_ends)] == expected_ends
    assert out.n_folds == len(seen_ends)


def test_min_variance_rule_runs_end_to_end() -> None:
    prices = _prices()

    def rule(train: pd.DataFrame) -> pd.Series:
        return min_variance_weights(train.pct_change().dropna())

    out = walk_forward_weights(prices, rule, train_window=80, test_window=20)
    equity = out.results["equity_curve"]
    assert np.isfinite(equity.to_numpy()).all()
    # fully invested from the first decision bar on; the final bar carries no
    # decision (nothing left to hold after the sample ends)
    assert (out.weights.iloc[80:-1].sum(axis=1) > 0.99).all()


def test_partial_final_fold_is_clipped() -> None:
    prices = _prices(75)
    out = walk_forward_weights(prices, _equal_weights, train_window=60, test_window=20)
    assert out.n_folds == 1
    assert len(out.weights) == 75


def test_bad_weight_fn_outputs_raise() -> None:
    prices = _prices(80)
    with pytest.raises(ValueError, match="unknown assets"):
        walk_forward_weights(
            prices, lambda t: pd.Series({"zzz": 1.0}), train_window=60, test_window=10
        )
    with pytest.raises(ValueError, match="non-finite"):
        walk_forward_weights(
            prices, lambda t: pd.Series({"aaa": np.nan}), train_window=60, test_window=10
        )


def test_bad_windows_raise() -> None:
    prices = _prices(80)
    with pytest.raises(ValueError, match="train_window"):
        walk_forward_weights(prices, _equal_weights, train_window=1)
    with pytest.raises(ValueError, match="test_window"):
        walk_forward_weights(prices, _equal_weights, train_window=60, test_window=0)
    with pytest.raises(ValueError, match="more than train_window"):
        walk_forward_weights(prices.iloc[:60], _equal_weights, train_window=60)
