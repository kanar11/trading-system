"""Tests for the execution / slippage model."""

import numpy as np
import pandas as pd
import pytest

from src.execution.slippage import (
    ExecutionConfig,
    compute_execution_cost,
    apply_execution_costs,
)
from src.backtest.engine import backtest_strategy


# ---------------------------------------------------------------------------
# compute_execution_cost
# ---------------------------------------------------------------------------

def test_no_trade_no_cost():
    cfg = ExecutionConfig(spread_bps=10, impact_coeff=0.5, fixed_cost_per_trade=0.001)
    assert compute_execution_cost(0.0, cfg) == 0.0


def test_spread_only_at_tiny_size():
    # impact_coeff=0 → only the half-spread component contributes
    cfg = ExecutionConfig(spread_bps=10, impact_coeff=0.0, fixed_cost_per_trade=0.0)
    # 10 bps round-trip → 5 bps half-spread = 0.0005
    assert compute_execution_cost(0.01, cfg) == pytest.approx(0.0005)


def test_impact_sqrt_law():
    # impact = coeff * (size / cap) ** 0.5 — quadrupling size doubles impact
    cfg = ExecutionConfig(
        spread_bps=0, impact_coeff=0.10, impact_exponent=0.5,
        participation_cap=1.0, fixed_cost_per_trade=0.0,
    )
    c1 = compute_execution_cost(0.04, cfg)
    c4 = compute_execution_cost(0.16, cfg)
    assert c4 == pytest.approx(2 * c1, rel=1e-9)


def test_vectorised_matches_scalar():
    cfg = ExecutionConfig(spread_bps=5, impact_coeff=0.1)
    sizes = np.array([0.0, 0.01, 0.05, 0.10])
    vec = compute_execution_cost(sizes, cfg)
    for size, expected in zip(sizes, vec):
        assert compute_execution_cost(float(size), cfg) == pytest.approx(expected)


def test_works_on_pandas_series():
    cfg = ExecutionConfig(spread_bps=5)
    s = pd.Series([0.0, 0.01, 0.05])
    out = compute_execution_cost(s.values, cfg)
    assert out.shape == s.shape
    assert (out >= 0).all()


# ---------------------------------------------------------------------------
# apply_execution_costs
# ---------------------------------------------------------------------------

def _engine_df() -> pd.DataFrame:
    dates = pd.date_range("2020-01-01", periods=20, freq="B")
    prices = [100 + i for i in range(20)]
    signals = [0, 1, 1, 1, 1, 0, -1, -1, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0]
    df = pd.DataFrame({"close": prices, "signal": signals}, index=dates)
    bt, _ = backtest_strategy(df, transaction_cost=0.0, vol_target=None)
    return bt


def test_apply_execution_costs_changes_cost_and_equity():
    bt = _engine_df()
    cfg = ExecutionConfig(spread_bps=20, impact_coeff=0.5, fixed_cost_per_trade=0.0001)
    out = apply_execution_costs(bt.copy(), cfg)

    # cost is replaced and strictly non-negative on trade bars
    assert (out["transaction_cost"] >= 0).all()
    # at least one bar must have a higher cost than the zero-cost baseline
    assert (out["transaction_cost"] > 0).any()
    # equity curve is recomputed
    assert "equity_curve" in out.columns


def test_apply_execution_costs_higher_impact_means_lower_equity():
    bt = _engine_df()
    cheap = apply_execution_costs(
        bt.copy(),
        ExecutionConfig(spread_bps=1, impact_coeff=0.01, fixed_cost_per_trade=0),
    )
    expensive = apply_execution_costs(
        bt.copy(),
        ExecutionConfig(spread_bps=50, impact_coeff=1.0, fixed_cost_per_trade=0.001),
    )
    assert expensive["equity_curve"].iloc[-1] < cheap["equity_curve"].iloc[-1]


def test_apply_execution_costs_raises_without_trade_col():
    df = pd.DataFrame({"strategy_returns_gross": [0.01, 0.02]})
    with pytest.raises(ValueError, match="trade"):
        apply_execution_costs(df)


def test_apply_execution_costs_raises_without_gross_returns():
    df = pd.DataFrame({"trade": [0.0, 0.1, 0.0]})
    with pytest.raises(ValueError, match="gross"):
        apply_execution_costs(df)
