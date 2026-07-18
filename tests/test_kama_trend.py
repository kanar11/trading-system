"""Tests for the KAMA adaptive trend strategy."""

import numpy as np
import pandas as pd
import pytest

from src.backtest import backtest_strategy, run_signal_event_backtest
from src.strategy.kama_trend import kama_trend_strategy


def _frame(step: float, n: int = 120, start: float = 100.0) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=n, freq="B")
    close = start * np.cumprod(np.full(n, 1.0 + step))
    return pd.DataFrame(
        {
            "open": np.concatenate([[start], close[:-1]]),
            "high": close * 1.002,
            "low": close * 0.998,
            "close": close,
        },
        index=idx,
    )


def test_uptrend_goes_long() -> None:
    out = kama_trend_strategy(_frame(0.003), er_period=10)
    assert "kama" in out.columns
    assert (out["signal"].iloc[-40:] == 1).all()


def test_downtrend_goes_short_or_flat() -> None:
    df = _frame(-0.003)
    short = kama_trend_strategy(df, er_period=10)
    assert (short["signal"].iloc[-40:] == -1).all()
    long_only = kama_trend_strategy(df, er_period=10, allow_short=False)
    assert set(long_only["signal"].unique()) <= {0, 1}
    assert (long_only["signal"].iloc[-40:] == 0).all()


def test_warm_up_is_flat() -> None:
    out = kama_trend_strategy(_frame(0.003), er_period=15)
    assert (out["signal"].iloc[:15] == 0).all()


def test_band_holds_state_through_wobble() -> None:
    # trend up, then a ±1% oscillation around the average: with 3% exit
    # hysteresis the long is held; the raw crossover flips repeatedly
    n = 120
    idx = pd.date_range("2024-01-01", periods=n, freq="B")
    trend = 100.0 * np.cumprod(np.full(80, 1.004))
    wobble = trend[-1] * (1.0 + 0.01 * np.sin(np.arange(40)))
    close = np.concatenate([trend, wobble])
    df = pd.DataFrame({"close": close}, index=idx)
    banded = kama_trend_strategy(df, er_period=10, band=0.03)
    raw = kama_trend_strategy(df, er_period=10, band=0.0)
    assert (banded["signal"].iloc[-30:] == 1).all()  # hysteresis holds the long
    assert (raw["signal"].iloc[-30:] != 1).any()  # unbanded flips at least once


def test_runs_through_both_engines() -> None:
    out = kama_trend_strategy(_frame(0.003))
    vector, _ = backtest_strategy(out.copy(), transaction_cost=0.0005)
    assert float(vector["equity_curve"].iloc[-1]) > 1.0
    event = run_signal_event_backtest(out)
    assert float(event.equity_curve.iloc[-1]) > 100_000.0


def test_bad_inputs_raise() -> None:
    df = _frame(0.001)
    with pytest.raises(ValueError, match="close"):
        kama_trend_strategy(df.drop(columns=["close"]))
    with pytest.raises(ValueError, match="band"):
        kama_trend_strategy(df, band=-0.01)
    with pytest.raises(ValueError, match="er_period"):
        kama_trend_strategy(df, er_period=0)
