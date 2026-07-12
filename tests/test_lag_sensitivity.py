"""Tests for execution-lag sensitivity analysis."""

import numpy as np
import pandas as pd
import pytest

from src.backtest import backtest_strategy, lag_sensitivity


def _oracle_frame(n: int = 400, seed: int = 1) -> pd.DataFrame:
    """Noisy prices with a one-bar-foresight signal: decays fast with lag."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2022-01-03", periods=n, freq="B")
    close = pd.Series(100.0 * np.cumprod(1 + rng.normal(0, 0.01, n)), index=idx)
    signal = np.sign(close.pct_change().shift(-1)).fillna(0)
    return pd.DataFrame({"close": close, "signal": signal}, index=idx)


def test_output_shape_and_index() -> None:
    table = lag_sensitivity(_oracle_frame(), lags=(0, 1, 2))
    assert list(table.index) == [0, 1, 2]
    assert list(table.columns) == [
        "ann_return",
        "ann_vol",
        "sharpe",
        "max_drawdown",
        "final_equity",
    ]


def test_oracle_edge_collapses_with_lag() -> None:
    table = lag_sensitivity(_oracle_frame(), lags=(0, 2, 4), transaction_cost=0.0)
    # perfect one-bar foresight is enormous at lag 0 and gone by lag 2+
    assert table.loc[0, "sharpe"] > 5
    assert table.loc[2, "sharpe"] < table.loc[0, "sharpe"] / 2
    assert table.loc[4, "sharpe"] < table.loc[0, "sharpe"] / 2


def test_lag_zero_matches_direct_backtest() -> None:
    df = _oracle_frame(200)
    table = lag_sensitivity(df, lags=(0,), transaction_cost=0.001)
    result, _ = backtest_strategy(df.copy(), transaction_cost=0.001)
    expected_final = float(result["equity_curve"].iloc[-1])
    assert table.loc[0, "final_equity"] == pytest.approx(expected_final)
    expected_ann = float(result["strategy_returns"].mean()) * 252
    assert table.loc[0, "ann_return"] == pytest.approx(expected_ann)


def test_constant_signal_is_lag_invariant() -> None:
    df = _oracle_frame(200)
    df["signal"] = 1  # buy & hold: delay changes only the first bars
    table = lag_sensitivity(df, lags=(0, 3), transaction_cost=0.0)
    assert table.loc[0, "final_equity"] == pytest.approx(table.loc[3, "final_equity"], rel=0.05)


def test_max_drawdown_is_non_positive() -> None:
    table = lag_sensitivity(_oracle_frame(), lags=(0, 1))
    assert (table["max_drawdown"] <= 0).all()


def test_flat_signal_gives_nan_sharpe() -> None:
    df = _oracle_frame(100)
    df["signal"] = 0
    table = lag_sensitivity(df, lags=(0,), transaction_cost=0.0)
    assert np.isnan(table.loc[0, "sharpe"])
    assert table.loc[0, "final_equity"] == pytest.approx(1.0)


def test_bad_lags_raise() -> None:
    df = _oracle_frame(50)
    with pytest.raises(ValueError, match="empty"):
        lag_sensitivity(df, lags=())
    with pytest.raises(ValueError, match=">= 0"):
        lag_sensitivity(df, lags=(0, -1))
    with pytest.raises(ValueError, match="periods_per_year"):
        lag_sensitivity(df, periods_per_year=0)
