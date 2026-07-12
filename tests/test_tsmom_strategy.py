"""Tests for the time-series momentum (TSMOM) strategy."""

import numpy as np
import pandas as pd
import pytest

from src.backtest import backtest_strategy
from src.data.calendar import rebalance_mask
from src.strategy.tsmom import tsmom_strategy


def _frame(step: float, n: int = 200, start: float = 100.0) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=n, freq="B")
    close = start * np.cumprod(np.full(n, 1.0 + step))
    return pd.DataFrame({"close": close}, index=idx)


def test_uptrend_goes_long_after_warm_up() -> None:
    out = tsmom_strategy(_frame(0.002), lookback=40, skip=5)
    assert "momentum_score" in out.columns
    assert (out["signal"].iloc[-40:] == 1).all()


def test_downtrend_goes_short() -> None:
    out = tsmom_strategy(_frame(-0.002), lookback=40, skip=5)
    assert (out["signal"].iloc[-40:] == -1).all()


def test_long_only_maps_short_to_flat() -> None:
    out = tsmom_strategy(_frame(-0.002), lookback=40, skip=5, allow_short=False)
    assert set(out["signal"].unique()) <= {0, 1}
    assert (out["signal"].iloc[-40:] == 0).all()


def test_flat_during_warm_up() -> None:
    out = tsmom_strategy(_frame(0.002), lookback=60, skip=10)
    assert (out["signal"].iloc[:60] == 0).all()


def test_signal_changes_only_on_rebalance_bars() -> None:
    df = _frame(0.002)
    out = tsmom_strategy(df, lookback=40, skip=5)
    changed = (out["signal"].diff().abs() > 0).to_numpy()
    month_ends = rebalance_mask(pd.DatetimeIndex(df.index), freq="M").to_numpy()
    assert not changed[~month_ends].any()


def test_skip_month_is_respected() -> None:
    # rally long past, crash in the last 10 bars: with skip=10 the score
    # ignores the crash and stays long at the next decision
    n = 120
    idx = pd.date_range("2024-01-01", periods=n, freq="B")
    close = np.concatenate(
        [100.0 * np.cumprod(np.full(n - 10, 1.004)), np.full(10, 50.0)]  # crash to 50
    )
    df = pd.DataFrame({"close": close}, index=idx)
    out = tsmom_strategy(df, lookback=60, skip=10)
    assert out["signal"].iloc[-1] == 1  # crash sits inside the skip window
    no_skip = tsmom_strategy(df, lookback=60, skip=0)
    assert no_skip["signal"].iloc[-1] == -1  # without skip the crash flips it


def test_runs_through_the_backtest_engine() -> None:
    out = tsmom_strategy(_frame(0.002), lookback=40, skip=5)
    result, _ = backtest_strategy(out, transaction_cost=0.0005)
    equity = result["equity_curve"]
    assert np.isfinite(equity.to_numpy()).all()
    assert equity.iloc[-1] > 1.0  # riding a clean up-trend is profitable


def test_missing_close_raises() -> None:
    idx = pd.date_range("2024-01-01", periods=5, freq="B")
    with pytest.raises(ValueError, match="close"):
        tsmom_strategy(pd.DataFrame({"open": np.ones(5)}, index=idx))
