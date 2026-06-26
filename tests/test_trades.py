"""Tests for the standalone 'pos'-convention trade log builder."""

import pandas as pd
import pytest

from quantbt.reporting.trades import build_trade_log


def _df(pos: list[int], close: list[float]) -> pd.DataFrame:
    dates = pd.date_range("2020-01-01", periods=len(pos), freq="D")
    return pd.DataFrame({"pos": pos, "close": close}, index=dates)


def test_missing_pos_column_raises() -> None:
    with pytest.raises(ValueError, match="'pos' column"):
        build_trade_log(pd.DataFrame({"close": [1.0, 2.0]}))


def test_missing_close_column_raises() -> None:
    with pytest.raises(ValueError, match="'close' column"):
        build_trade_log(pd.DataFrame({"pos": [0, 1]}))


def test_no_positions_returns_empty() -> None:
    result = build_trade_log(_df([0, 0, 0], [100.0, 101.0, 102.0]))
    assert result.empty


def test_single_long_trade_return() -> None:
    df = _df([0, 1, 1, 0], [100.0, 100.0, 110.0, 110.0])
    log = build_trade_log(df)
    assert len(log) == 1
    row = log.iloc[0]
    assert row["direction"] == 1
    assert row["trade_return"] == pytest.approx(0.10)
    assert row["holding_days"] == 2


def test_short_trade_profits_when_price_falls() -> None:
    df = _df([0, -1, -1, 0], [100.0, 100.0, 90.0, 90.0])
    log = build_trade_log(df)
    assert len(log) == 1
    assert log.iloc[0]["direction"] == -1
    assert log.iloc[0]["trade_return"] == pytest.approx(0.10)


def test_open_position_marked_to_market_at_end() -> None:
    df = _df([0, 1, 1], [100.0, 100.0, 105.0])
    log = build_trade_log(df)
    assert len(log) == 1
    assert log.iloc[0]["exit_price"] == pytest.approx(105.0)


def test_position_flip_creates_two_trades() -> None:
    df = _df([1, 1, -1, -1], [100.0, 110.0, 110.0, 100.0])
    log = build_trade_log(df)
    assert len(log) == 2
    assert list(log["direction"]) == [1, -1]
