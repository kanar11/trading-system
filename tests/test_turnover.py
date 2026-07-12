"""Tests for turnover measurement and turnover-constrained rebalancing."""

import numpy as np
import pandas as pd
import pytest

from src.portfolio import constrain_turnover, portfolio_turnover


def test_turnover_is_sum_of_absolute_deltas() -> None:
    current = pd.Series({"a": 0.6, "b": 0.4})
    target = pd.Series({"a": 0.4, "b": 0.6})
    assert portfolio_turnover(current, target) == pytest.approx(0.4)


def test_turnover_counts_entries_and_exits() -> None:
    current = pd.Series({"a": 1.0})
    target = pd.Series({"b": 1.0})
    # full exit (1.0) + full entry (1.0)
    assert portfolio_turnover(current, target) == pytest.approx(2.0)


def test_within_budget_returns_target_unchanged() -> None:
    current = pd.Series({"a": 0.5, "b": 0.5})
    target = pd.Series({"a": 0.55, "b": 0.45})
    out = constrain_turnover(current, target, max_turnover=0.5)
    assert out.name == "weights"
    assert np.allclose(out.to_numpy(), target.to_numpy())


def test_over_budget_spends_exactly_the_budget() -> None:
    current = pd.Series({"a": 0.8, "b": 0.2})
    target = pd.Series({"a": 0.2, "b": 0.8})  # turnover 1.2
    out = constrain_turnover(current, target, max_turnover=0.3)
    assert portfolio_turnover(current, out) == pytest.approx(0.3)
    # each position moved a quarter of the way (0.3 / 1.2)
    assert out["a"] == pytest.approx(0.8 - 0.6 * 0.25)
    assert out["b"] == pytest.approx(0.2 + 0.6 * 0.25)


def test_result_lies_between_current_and_target() -> None:
    current = pd.Series({"a": 0.7, "b": 0.3, "c": 0.0})
    target = pd.Series({"a": 0.1, "b": 0.4, "c": 0.5})
    out = constrain_turnover(current, target, max_turnover=0.4)
    lower = np.minimum(current.to_numpy(), target.reindex(current.index).to_numpy())
    upper = np.maximum(current.to_numpy(), target.reindex(current.index).to_numpy())
    values = out.reindex(current.index).to_numpy()
    assert (values >= lower - 1e-12).all()
    assert (values <= upper + 1e-12).all()


def test_weight_sum_is_preserved() -> None:
    current = pd.Series({"a": 0.6, "b": 0.4})
    target = pd.Series({"a": 0.1, "b": 0.5, "c": 0.4})
    out = constrain_turnover(current, target, max_turnover=0.2)
    assert float(out.sum()) == pytest.approx(1.0)


def test_zero_budget_keeps_current_book() -> None:
    current = pd.Series({"a": 0.6, "b": 0.4})
    target = pd.Series({"a": 0.0, "b": 1.0})
    out = constrain_turnover(current, target, max_turnover=0.0)
    assert np.allclose(out.reindex(current.index).to_numpy(), current.to_numpy())


def test_identical_books_need_no_trading() -> None:
    weights = pd.Series({"a": 0.5, "b": 0.5})
    assert portfolio_turnover(weights, weights) == 0.0
    out = constrain_turnover(weights, weights, max_turnover=0.0)
    assert np.allclose(out.to_numpy(), weights.to_numpy())


def test_bad_inputs_raise() -> None:
    weights = pd.Series({"a": 0.5, "b": 0.5})
    with pytest.raises(ValueError, match="max_turnover"):
        constrain_turnover(weights, weights, max_turnover=-0.1)
    with pytest.raises(ValueError, match="finite"):
        portfolio_turnover(weights, pd.Series({"a": np.nan, "b": 0.5}))
