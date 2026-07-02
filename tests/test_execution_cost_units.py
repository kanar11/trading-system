"""Regression tests: execution costs must scale with traded notional.

The old ``apply_execution_costs`` subtracted the *per-unit-of-notional*
cost fraction straight from equity returns, so a 0.001 rebalance paid
exactly the same spread as a full 2.0 position flip. The cost charged
to returns must be ``per-unit cost x turnover``, matching how the
engine applies its flat ``transaction_cost``.
"""

import numpy as np
import pandas as pd
import pytest

from src.backtest.engine import backtest_strategy
from src.execution.slippage import ExecutionConfig, apply_execution_costs


def _bt_frame(trades: list[float]) -> pd.DataFrame:
    """Minimal frame with the columns apply_execution_costs requires."""
    index = pd.date_range("2021-01-04", periods=len(trades), freq="B")
    return pd.DataFrame(
        {"trade": trades, "strategy_returns_gross": [0.0] * len(trades)},
        index=index,
    )


def test_spread_cost_is_linear_in_turnover():
    cfg = ExecutionConfig(spread_bps=10, impact_coeff=0.0, fixed_cost_per_trade=0.0)
    out = apply_execution_costs(_bt_frame([0.0, 0.5, 1.0, 2.0]), cfg)

    half_spread = 0.5 * 10 / 10_000.0
    expected = np.array([0.0, 0.5, 1.0, 2.0]) * half_spread
    assert np.allclose(out["transaction_cost"].to_numpy(), expected)


def test_small_rebalance_pays_less_than_full_flip():
    cfg = ExecutionConfig(spread_bps=10, impact_coeff=0.0, fixed_cost_per_trade=0.0)
    out = apply_execution_costs(_bt_frame([0.001, 2.0]), cfg)

    small, big = out["transaction_cost"].to_numpy()
    assert big == pytest.approx(small * 2000.0)


def test_sqrt_impact_charge_scales_superlinearly():
    # per-unit impact ~ size^0.5, so the charge ~ size^1.5: 4x size → 8x cost
    cfg = ExecutionConfig(
        spread_bps=0.0,
        impact_coeff=0.1,
        impact_exponent=0.5,
        participation_cap=1.0,
        fixed_cost_per_trade=0.0,
    )
    out = apply_execution_costs(_bt_frame([0.04, 0.16]), cfg)

    c_small, c_big = out["transaction_cost"].to_numpy()
    assert c_big == pytest.approx(8.0 * c_small, rel=1e-9)


def test_zero_trade_bars_pay_nothing():
    cfg = ExecutionConfig(spread_bps=25, impact_coeff=0.5, fixed_cost_per_trade=0.001)
    out = apply_execution_costs(_bt_frame([0.0, 0.0, 1.0, 0.0]), cfg)

    costs = out["transaction_cost"].to_numpy()
    assert costs[0] == 0.0
    assert costs[1] == 0.0
    assert costs[2] > 0.0
    assert costs[3] == 0.0


def test_matches_engine_flat_model_when_only_spread_is_active():
    """With impact and fixed costs off, the model must reproduce the
    engine's flat per-turnover cost exactly."""
    dates = pd.date_range("2021-01-04", periods=12, freq="B")
    df = pd.DataFrame(
        {
            "close": [100 + i for i in range(12)],
            "signal": [0, 1, 1, 0, -1, -1, 0, 1, 1, 1, 0, 0],
        },
        index=dates,
    )
    half_spread = 0.0007

    flat_bt, _ = backtest_strategy(df.copy(), transaction_cost=half_spread)
    zero_bt, _ = backtest_strategy(df.copy(), transaction_cost=0.0)
    cfg = ExecutionConfig(
        spread_bps=half_spread * 2 * 10_000.0,
        impact_coeff=0.0,
        fixed_cost_per_trade=0.0,
    )
    modelled = apply_execution_costs(zero_bt, cfg)

    assert np.allclose(
        modelled["transaction_cost"].to_numpy(),
        flat_bt["transaction_cost"].to_numpy(),
    )
    assert np.allclose(
        modelled["equity_curve"].to_numpy(),
        flat_bt["equity_curve"].to_numpy(),
    )
