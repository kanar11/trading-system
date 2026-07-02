"""Regression tests: risk controls must be free of look-ahead bias.

Every rule is decided on a bar's close and executed on the *next* bar
(the engine's ``signal.shift(1)`` convention). The old implementation
zeroed the position on the decision bar itself, retroactively deleting
the very loss that triggered the rule and inflating backtest results.
"""

import numpy as np
import pandas as pd

from src.backtest.engine import backtest_strategy
from src.risk.manager import RiskConfig, apply_risk_controls


def _risk_df(prices: list[float], positions: list[float], index=None) -> pd.DataFrame:
    """Minimal risk-manager input frame."""
    if index is None:
        index = pd.date_range("2020-01-01", periods=len(prices), freq="B")
    return pd.DataFrame(
        {"close": prices, "position": positions, "scaled_position": positions},
        index=index,
    )


def test_stop_loss_decision_bar_keeps_its_loss_in_the_engine():
    """The bar whose close breaches the stop must still book that bar's P&L."""
    dates = pd.date_range("2020-01-01", periods=6, freq="B")
    df = pd.DataFrame(
        {"close": [100, 100, 100, 90, 89, 88], "signal": [1, 1, 1, 1, 1, 1]},
        index=dates,
    )
    config = RiskConfig(stop_loss=0.05, take_profit=None, trailing_stop=None, daily_loss_limit=None)
    managed, _ = backtest_strategy(df.copy(), transaction_cost=0.0, risk_config=config)
    unmanaged, _ = backtest_strategy(df.copy(), transaction_cost=0.0)

    # decision bar = index 3 (close 90, -10% from entry at 100)
    assert managed["risk_event"].iloc[3] == "stop_loss"
    # its return equals the unmanaged return — the loss is NOT erased
    assert managed["strategy_returns"].iloc[3] == unmanaged["strategy_returns"].iloc[3]
    assert managed["strategy_returns"].iloc[3] < 0
    # the exit takes effect on the following bar
    assert managed["scaled_position"].iloc[4] == 0.0
    assert managed["strategy_returns"].iloc[4] == 0.0


def test_risk_managed_equity_cannot_beat_unmanaged_on_the_decision_bar():
    """Up to and including the decision bar the two equity curves match."""
    dates = pd.date_range("2020-01-01", periods=5, freq="B")
    df = pd.DataFrame(
        {"close": [100, 100, 93, 92, 91], "signal": [1, 1, 1, 1, 1]},
        index=dates,
    )
    config = RiskConfig(stop_loss=0.05, take_profit=None, trailing_stop=None, daily_loss_limit=None)
    managed, _ = backtest_strategy(df.copy(), transaction_cost=0.0, risk_config=config)
    unmanaged, _ = backtest_strategy(df.copy(), transaction_cost=0.0)

    decision = 2  # close 93 → -7% from entry
    assert managed["risk_event"].iloc[decision] == "stop_loss"
    assert np.allclose(
        managed["equity_curve"].iloc[: decision + 1].to_numpy(),
        unmanaged["equity_curve"].iloc[: decision + 1].to_numpy(),
    )


def test_reentry_after_forced_exit_uses_fresh_entry_price():
    prices = [100, 100, 94, 93, 85, 84]
    positions = [0, 1, 1, 1, 1, 1]
    df = _risk_df(prices, positions)
    config = RiskConfig(stop_loss=0.05, take_profit=None, trailing_stop=None, daily_loss_limit=None)
    result = apply_risk_controls(df, config)

    # first stop: decided on bar 2 (94 vs entry 100), flat on bar 3
    assert result["risk_event"].iloc[2] == "stop_loss"
    assert result["scaled_position"].iloc[3] == 0.0
    # re-entry on bar 4 at the previous close (93); 85 breaches again
    assert result["scaled_position"].iloc[4] == 1.0
    assert result["risk_event"].iloc[4] == "stop_loss"
    assert result["scaled_position"].iloc[5] == 0.0


def test_exit_event_takes_precedence_over_pos_capped_flag():
    prices = [100, 100, 90, 90]
    positions = [0, 2.5, 2.5, 2.5]
    df = _risk_df(prices, positions)
    config = RiskConfig(
        stop_loss=0.05,
        take_profit=None,
        trailing_stop=None,
        max_position=1.5,
        daily_loss_limit=None,
    )
    result = apply_risk_controls(df, config)

    assert result["risk_event"].iloc[1] == "pos_capped"
    assert result["scaled_position"].iloc[1] == 1.5
    # bar 2: capped AND stop decided — exit label wins, cap still applied
    assert result["risk_event"].iloc[2] == "stop_loss"
    assert result["scaled_position"].iloc[2] == 1.5
    assert result["scaled_position"].iloc[3] == 0.0


def test_daily_loss_limit_accrues_pnl_with_position_held_over_the_bar():
    # intraday bars on a single calendar day
    index = pd.date_range("2020-01-06 09:00", periods=6, freq="h")
    prices = [100, 100, 97, 96, 95, 94]
    positions = [0, 1, 1, 1, 1, 1]
    df = _risk_df(prices, positions, index=index)
    config = RiskConfig(stop_loss=None, take_profit=None, trailing_stop=None, daily_loss_limit=0.02)
    result = apply_risk_controls(df, config)

    # bar 2 loses 3% with a full position → breach decided there
    assert result["risk_event"].iloc[2] == "daily_limit"
    assert result["scaled_position"].iloc[2] == 1.0  # decision bar keeps the loss
    # flat from the next bar for the rest of the day
    assert result["scaled_position"].iloc[3] == 0.0
    assert result["scaled_position"].iloc[4] == 0.0
    assert result["risk_event"].iloc[4] == "daily_limit"
    assert result["scaled_position"].iloc[5] == 0.0


def test_circuit_breaker_releases_on_the_next_day():
    day1 = pd.date_range("2020-01-06 09:00", periods=3, freq="h")
    day2 = pd.date_range("2020-01-07 09:00", periods=3, freq="h")
    index = day1.append(day2)
    prices = [100, 96, 95, 95, 96, 97]
    positions = [1, 1, 1, 1, 1, 1]
    df = _risk_df(prices, positions, index=index)
    config = RiskConfig(stop_loss=None, take_profit=None, trailing_stop=None, daily_loss_limit=0.02)
    result = apply_risk_controls(df, config)

    # -4% on day 1 bar 1 → breach; bar 2 is the forced-flat execution bar
    assert result["risk_event"].iloc[1] == "daily_limit"
    assert result["scaled_position"].iloc[2] == 0.0
    # day 2: breaker released, position resumes with no daily_limit flag
    assert result["scaled_position"].iloc[3] == 1.0
    assert result["risk_event"].iloc[3] == ""
