"""Tests for the signal-to-event-engine bridge."""

import numpy as np
import pandas as pd
import pytest

from src.backtest import backtest_strategy, run_signal_event_backtest
from src.oms import OrderType, TimeInForce


def _frame(n: int = 60, step: float = 0.003, signal: int = 1) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=n, freq="B")
    close = 100.0 * np.cumprod(np.full(n, 1.0 + step))
    return pd.DataFrame(
        {
            "open": np.concatenate([[100.0], close[:-1]]),  # opens at prior close
            "high": close * 1.002,
            "low": close * 0.997,
            "close": close,
            "signal": signal,
        },
        index=idx,
    )


def test_constant_long_signal_enters_once() -> None:
    result = run_signal_event_backtest(_frame())
    filled = [o for o in result.orders if o.status.name == "FILLED"]
    assert len(filled) == 1
    assert filled[0].order_type is OrderType.MARKET
    assert filled[0].time_in_force is TimeInForce.GTC


def test_frictionless_equity_matches_hand_computation() -> None:
    df = _frame()
    result = run_signal_event_backtest(df)
    # decision on bar 0's close, filled at bar 1's open with qty = equity/close0
    qty = 100_000.0 / float(df["close"].iloc[0])
    expected = 100_000.0 - qty * float(df["open"].iloc[1]) + qty * float(df["close"].iloc[-1])
    assert float(result.equity_curve.iloc[-1]) == pytest.approx(expected)


def test_signal_flip_trades_through_zero() -> None:
    df = _frame()
    df.loc[df.index[30:], "signal"] = -1
    result = run_signal_event_backtest(df)
    filled = [o for o in result.orders if o.status.name == "FILLED"]
    assert len(filled) == 2
    # the flip order closes the long and opens the short in one delta
    long_qty = filled[0].quantity
    assert filled[1].quantity > long_qty


def test_flat_signal_never_trades() -> None:
    result = run_signal_event_backtest(_frame(signal=0))
    assert len(result.orders) == 0
    assert np.allclose(result.equity_curve.to_numpy(), 100_000.0)


def test_nan_signal_is_flat() -> None:
    df = _frame()
    df["signal"] = np.nan
    result = run_signal_event_backtest(df)
    assert len(result.orders) == 0


def test_costs_reduce_final_equity() -> None:
    df = _frame()
    free = run_signal_event_backtest(df)
    costly = run_signal_event_backtest(
        df, commission_per_share=0.01, commission_min=1.0, slippage_bps=10.0
    )
    assert float(costly.equity_curve.iloc[-1]) < float(free.equity_curve.iloc[-1])
    assert costly.portfolio.fees_paid > 0


def test_direction_agrees_with_vectorised_engine() -> None:
    df = _frame()
    event = run_signal_event_backtest(df)
    vector, _ = backtest_strategy(df.copy(), transaction_cost=0.0)
    assert float(event.equity_curve.iloc[-1]) > 100_000.0
    assert float(vector["equity_curve"].iloc[-1]) > 1.0


def test_position_fraction_scales_exposure() -> None:
    df = _frame()
    half = run_signal_event_backtest(df, position_fraction=0.5)
    full = run_signal_event_backtest(df, position_fraction=1.0)
    half_gain = float(half.equity_curve.iloc[-1]) - 100_000.0
    full_gain = float(full.equity_curve.iloc[-1]) - 100_000.0
    assert half_gain == pytest.approx(full_gain / 2, rel=1e-9)


def test_bad_inputs_raise() -> None:
    df = _frame()
    with pytest.raises(ValueError, match="signal"):
        run_signal_event_backtest(df.drop(columns=["signal"]))
    with pytest.raises(ValueError, match="position_fraction"):
        run_signal_event_backtest(df, position_fraction=0.0)
