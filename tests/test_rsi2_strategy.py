"""Tests for the Connors RSI-2 pullback strategy."""

import numpy as np
import pandas as pd
import pytest

from src.backtest import backtest_strategy, run_signal_event_backtest
from src.strategy.rsi2 import rsi2_strategy


def _uptrend_with_pullback(n: int = 260) -> pd.DataFrame:
    """A long, steady up-trend with one sharp dip late in the series."""
    idx = pd.date_range("2023-01-02", periods=n, freq="B")
    close = 100.0 + 0.3 * np.arange(n, dtype=float)
    close[240:245] -= 8.0  # a short, sharp pullback that recovers
    return pd.DataFrame(
        {
            "open": close,
            "high": close + 0.5,
            "low": close - 0.5,
            "close": close,
        },
        index=idx,
    )


def test_columns_present() -> None:
    out = rsi2_strategy(_uptrend_with_pullback())
    for column in ("rsi", "sma_trend", "sma_exit", "signal"):
        assert column in out.columns


def test_buys_the_pullback_in_an_uptrend() -> None:
    out = rsi2_strategy(_uptrend_with_pullback(), trend_window=200, entry_threshold=10.0)
    # the dip drives RSI(2) to near-zero while price stays above SMA-200 -> long
    pullback = out["signal"].iloc[241:246]
    assert (pullback == 1).any()
    assert (out["signal"] >= 0).all()  # never short in an up-trend


def test_warm_up_is_flat() -> None:
    out = rsi2_strategy(_uptrend_with_pullback(), trend_window=200)
    assert (out["signal"].iloc[:199] == 0).all()  # SMA-200 not yet defined


def test_exits_when_price_recovers_above_exit_ma() -> None:
    out = rsi2_strategy(_uptrend_with_pullback(), trend_window=200, exit_window=5)
    # once the dip recovers and price is back above SMA-5 the position closes
    assert out["signal"].iloc[-1] == 0


def test_downtrend_shorts_the_bounce() -> None:
    n = 260
    idx = pd.date_range("2023-01-02", periods=n, freq="B")
    close = 200.0 - 0.3 * np.arange(n, dtype=float)
    close[240:245] += 8.0  # a sharp bounce inside the down-trend
    df = pd.DataFrame(
        {"open": close, "high": close + 0.5, "low": close - 0.5, "close": close},
        index=idx,
    )
    out = rsi2_strategy(df, trend_window=200, entry_threshold=10.0)
    assert (out["signal"].iloc[241:246] == -1).any()
    assert (out["signal"] <= 0).all()  # never long in a down-trend


def test_long_only_clamps_shorts() -> None:
    n = 260
    idx = pd.date_range("2023-01-02", periods=n, freq="B")
    close = 200.0 - 0.3 * np.arange(n, dtype=float)
    close[240:245] += 8.0
    df = pd.DataFrame(
        {"open": close, "high": close + 0.5, "low": close - 0.5, "close": close},
        index=idx,
    )
    out = rsi2_strategy(df, allow_short=False)
    assert set(out["signal"].unique()) <= {0, 1}


def test_runs_through_both_engines() -> None:
    out = rsi2_strategy(_uptrend_with_pullback())
    vector, _ = backtest_strategy(out.copy(), transaction_cost=0.0005)
    assert np.isfinite(vector["equity_curve"].to_numpy()).all()
    event = run_signal_event_backtest(out)
    assert np.isfinite(event.equity_curve.to_numpy()).all()


def test_bad_inputs_raise() -> None:
    df = _uptrend_with_pullback(50)
    with pytest.raises(ValueError, match="close"):
        rsi2_strategy(df.drop(columns=["close"]))
    with pytest.raises(ValueError, match="entry_threshold"):
        rsi2_strategy(df, entry_threshold=0.0)
    with pytest.raises(ValueError, match="entry_threshold"):
        rsi2_strategy(df, entry_threshold=60.0)
