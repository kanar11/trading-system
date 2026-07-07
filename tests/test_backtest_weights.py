"""Tests for the multi-asset weight-frame backtest engine."""

import numpy as np
import pandas as pd
import pytest

from src.backtest import backtest_weights
from src.strategy.dual_momentum import dual_momentum_strategy


def _prices(n: int = 120) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=n, freq="B")
    up = 100.0 * np.cumprod(np.full(n, 1.004))
    down = 100.0 * np.cumprod(np.full(n, 0.996))
    return pd.DataFrame({"up": up, "down": down}, index=idx)


def _full_weight(prices: pd.DataFrame, column: str) -> pd.DataFrame:
    weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    weights[column] = 1.0
    return weights


def test_constant_full_weight_reproduces_buy_and_hold() -> None:
    prices = _prices()
    result = backtest_weights(prices, _full_weight(prices, "up"), cost_bps=0.0)
    expected = float(prices["up"].iloc[-1] / prices["up"].iloc[0])
    assert result["equity_curve"].iloc[-1] == pytest.approx(expected)


def test_output_shape_and_columns() -> None:
    prices = _prices(30)
    result = backtest_weights(prices, _full_weight(prices, "up"))
    assert list(result.columns) == [
        "portfolio_return_gross",
        "turnover",
        "cost",
        "portfolio_return",
        "equity_curve",
    ]
    assert result.index.equals(prices.index)


def test_entry_turnover_charged_once() -> None:
    prices = _prices(10)
    result = backtest_weights(prices, _full_weight(prices, "up"), cost_bps=10.0)
    # weights decided at bar 0 are held from bar 1: single entry trade
    assert result["turnover"].iloc[0] == 0.0
    assert result["turnover"].iloc[1] == 1.0
    assert (result["turnover"].iloc[2:] == 0.0).all()
    assert result["cost"].iloc[1] == pytest.approx(0.001)


def test_flip_charges_double_turnover() -> None:
    prices = _prices(10)
    weights = _full_weight(prices, "up")
    weights.iloc[5:] = 0.0
    weights.iloc[5:, list(prices.columns).index("down")] = -1.0  # flip long->short
    result = backtest_weights(prices, weights, cost_bps=10.0)
    # held changes at bar 6: up 1->0 and down 0->-1 -> turnover 2
    assert result["turnover"].iloc[6] == pytest.approx(2.0)


def test_zero_weights_stay_flat() -> None:
    prices = _prices(20)
    weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    result = backtest_weights(prices, weights)
    assert (result["equity_curve"] == 1.0).all()
    assert (result["cost"] == 0.0).all()


def test_costs_reduce_equity() -> None:
    prices = _prices()
    weights = _full_weight(prices, "up")
    free = backtest_weights(prices, weights, cost_bps=0.0)
    costly = backtest_weights(prices, weights, cost_bps=50.0)
    assert costly["equity_curve"].iloc[-1] < free["equity_curve"].iloc[-1]


def test_dual_momentum_runs_end_to_end() -> None:
    prices = _prices(130)
    weights = dual_momentum_strategy(prices, lookback=20, top_n=1)
    result = backtest_weights(prices, weights, cost_bps=10.0)
    equity = result["equity_curve"]
    assert np.isfinite(equity.to_numpy()).all()
    # riding the +0.4%/bar asset must end profitably even after costs
    assert equity.iloc[-1] > 1.0


def test_misaligned_frames_raise() -> None:
    prices = _prices(10)
    weights = _full_weight(prices, "up")
    with pytest.raises(ValueError, match="index"):
        backtest_weights(prices.iloc[:-1], weights)
    with pytest.raises(ValueError, match="columns"):
        backtest_weights(prices, weights[["down", "up"]])


def test_bad_inputs_raise() -> None:
    prices = _prices(10)
    weights = _full_weight(prices, "up")
    with pytest.raises(ValueError, match="cost_bps"):
        backtest_weights(prices, weights, cost_bps=-1.0)
    bad_weights = weights.copy()
    bad_weights.iloc[3, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        backtest_weights(prices, bad_weights)
    bad_prices = prices.copy()
    bad_prices.iloc[2, 0] = -5.0
    with pytest.raises(ValueError, match="positive"):
        backtest_weights(bad_prices, weights)
