"""Tests for per-trade MAE/MFE excursion analysis."""

import numpy as np
import pandas as pd
import pytest

from src.backtest import backtest_strategy, trade_excursions


def _ohlc(closes: list[float], spread: float = 0.0) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=len(closes), freq="B")
    close = pd.Series(closes, index=idx, dtype=float)
    return pd.DataFrame(
        {
            "open": close,
            "high": close + spread,
            "low": close - spread,
            "close": close,
        },
        index=idx,
    )


def _log(entry: str, exit_: str, direction: int, entry_price: float, ret: float) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "entry_date": pd.Timestamp(entry),
                "exit_date": pd.Timestamp(exit_),
                "direction": direction,
                "entry_price": entry_price,
                "trade_return": ret,
            }
        ]
    )


def test_long_trade_mae_mfe_by_hand() -> None:
    # path: 100 -> 110 (high) -> 95 (low) -> 105 exit
    df = _ohlc([100.0, 110.0, 95.0, 105.0])
    log = _log("2024-01-01", "2024-01-04", 1, 100.0, 0.05)
    out = trade_excursions(df, log)
    assert out["mfe"].iloc[0] == pytest.approx(0.10)
    assert out["mae"].iloc[0] == pytest.approx(-0.05)
    assert out["efficiency"].iloc[0] == pytest.approx(0.5)


def test_short_trade_is_side_aware() -> None:
    # short from 100: low 90 is the favorable side, high 108 the adverse one
    df = _ohlc([100.0, 108.0, 90.0, 95.0])
    log = _log("2024-01-01", "2024-01-04", -1, 100.0, 100.0 / 95.0 - 1)
    out = trade_excursions(df, log)
    assert out["mfe"].iloc[0] == pytest.approx(100.0 / 90.0 - 1)
    assert out["mae"].iloc[0] == pytest.approx(100.0 / 108.0 - 1)
    assert out["mae"].iloc[0] < 0 < out["mfe"].iloc[0]


def test_never_adverse_trade_has_zero_mae() -> None:
    df = _ohlc([100.0, 104.0, 108.0, 112.0])
    log = _log("2024-01-01", "2024-01-04", 1, 100.0, 0.12)
    out = trade_excursions(df, log)
    assert out["mae"].iloc[0] == 0.0
    assert out["efficiency"].iloc[0] == pytest.approx(1.0)  # rode it to the top


def test_window_is_restricted_to_the_trade() -> None:
    # the crash AFTER the exit must not contaminate the trade's MAE
    df = _ohlc([100.0, 105.0, 103.0, 50.0])
    log = _log("2024-01-01", "2024-01-03", 1, 100.0, 0.03)
    out = trade_excursions(df, log)
    assert out["mae"].iloc[0] == pytest.approx(0.0)
    assert out["mfe"].iloc[0] == pytest.approx(0.05)


def test_empty_log_returns_empty_with_columns() -> None:
    df = _ohlc([100.0, 101.0])
    empty = pd.DataFrame(columns=["entry_date", "exit_date", "direction", "entry_price"])
    out = trade_excursions(df, empty)
    assert len(out) == 0
    assert {"mae", "mfe", "efficiency"} <= set(out.columns)


def test_integrates_with_the_engine_trade_log() -> None:
    rng = np.random.default_rng(5)
    closes = 100.0 * np.cumprod(1 + rng.normal(0, 0.01, 120))
    df = _ohlc(list(closes), spread=0.5)
    df["signal"] = np.where(np.arange(120) % 40 < 20, 1, -1)  # periodic flips
    result, trade_log = backtest_strategy(df.copy(), transaction_cost=0.0)
    assert len(trade_log) > 1
    out = trade_excursions(df, trade_log)
    assert len(out) == len(trade_log)
    assert (out["mae"] <= 0).all()
    assert (out["mfe"] >= 0).all()
    # realised return can never beat the best open profit
    realised = out["trade_return"].to_numpy(dtype=float)
    assert (realised <= out["mfe"].to_numpy() + 1e-12).all()


def test_missing_columns_raise() -> None:
    df = _ohlc([100.0, 101.0])
    log = _log("2024-01-01", "2024-01-02", 1, 100.0, 0.01)
    with pytest.raises(ValueError, match="high"):
        trade_excursions(df.drop(columns=["high"]), log)
    with pytest.raises(ValueError, match="entry_price"):
        trade_excursions(df, log.drop(columns=["entry_price"]))


def test_out_of_range_trade_raises() -> None:
    df = _ohlc([100.0, 101.0])
    log = _log("2030-01-01", "2030-01-05", 1, 100.0, 0.0)
    with pytest.raises(ValueError, match="no bars"):
        trade_excursions(df, log)
