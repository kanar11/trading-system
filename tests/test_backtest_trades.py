"""Tests for round-trip trade statistics."""

import math

import numpy as np
import pandas as pd
import pytest

from src.backtest import backtest_strategy, trade_statistics


def _log(returns: list[float], holding: list[int] | None = None) -> pd.DataFrame:
    data: dict[str, list[float] | list[int]] = {"trade_return": returns}
    if holding is not None:
        data["holding_days"] = holding
    return pd.DataFrame(data)


def test_basic_counts_and_rates() -> None:
    stats = trade_statistics(_log([0.10, -0.05, 0.20, -0.10]))
    assert stats.n_trades == 4
    assert stats.n_wins == 2
    assert stats.n_losses == 2
    assert stats.win_rate == pytest.approx(0.5)
    assert stats.avg_return == pytest.approx((0.10 - 0.05 + 0.20 - 0.10) / 4)


def test_win_loss_averages_and_extremes() -> None:
    stats = trade_statistics(_log([0.10, -0.05, 0.20, -0.10]))
    assert stats.avg_win == pytest.approx(0.15)
    assert stats.avg_loss == pytest.approx(-0.075)
    assert stats.best_trade == pytest.approx(0.20)
    assert stats.worst_trade == pytest.approx(-0.10)


def test_profit_factor_and_payoff() -> None:
    stats = trade_statistics(_log([0.10, -0.05, 0.20, -0.10]))
    assert stats.gross_profit == pytest.approx(0.30)
    assert stats.gross_loss == pytest.approx(0.15)
    assert stats.profit_factor == pytest.approx(2.0)
    assert stats.payoff_ratio == pytest.approx(0.15 / 0.075)


def test_all_wins_gives_infinite_ratios() -> None:
    stats = trade_statistics(_log([0.10, 0.05, 0.20]))
    assert stats.n_losses == 0
    assert math.isinf(stats.profit_factor)
    assert math.isinf(stats.payoff_ratio)
    assert stats.win_rate == pytest.approx(1.0)


def test_all_losses_gives_zero_ratios() -> None:
    stats = trade_statistics(_log([-0.10, -0.05]))
    assert stats.profit_factor == 0.0
    assert stats.payoff_ratio == 0.0
    assert stats.win_rate == 0.0


def test_consecutive_streaks() -> None:
    # W W L W L L L W
    stats = trade_statistics(_log([0.1, 0.2, -0.1, 0.3, -0.1, -0.2, -0.3, 0.4]))
    assert stats.max_consecutive_wins == 2
    assert stats.max_consecutive_losses == 3


def test_breakeven_trade_is_neither_win_nor_loss() -> None:
    stats = trade_statistics(_log([0.1, 0.0, -0.1]))
    assert stats.n_trades == 3
    assert stats.n_wins == 1
    assert stats.n_losses == 1
    # a flat trade breaks both streaks
    assert stats.max_consecutive_wins == 1
    assert stats.max_consecutive_losses == 1


def test_avg_holding_days() -> None:
    stats = trade_statistics(_log([0.1, -0.1], holding=[5, 15]))
    assert stats.avg_holding_days == pytest.approx(10.0)


def test_holding_column_absent_defaults_to_zero() -> None:
    stats = trade_statistics(_log([0.1, -0.1]))
    assert stats.avg_holding_days == 0.0


def test_empty_log_returns_zeroed_stats() -> None:
    stats = trade_statistics(pd.DataFrame())
    assert stats.n_trades == 0
    assert stats.win_rate == 0.0
    assert stats.profit_factor == 0.0
    assert stats.avg_holding_days == 0.0


def test_missing_return_column_raises() -> None:
    with pytest.raises(ValueError, match="trade_return"):
        trade_statistics(pd.DataFrame({"pnl": [0.1, -0.1]}))


def test_custom_return_column() -> None:
    stats = trade_statistics(pd.DataFrame({"ret": [0.1, -0.05]}), return_col="ret")
    assert stats.n_trades == 2
    assert stats.avg_return == pytest.approx(0.025)


def test_nan_returns_are_dropped() -> None:
    stats = trade_statistics(_log([0.1, float("nan"), -0.05]))
    assert stats.n_trades == 2


def test_all_nan_log_returns_zeroed_stats() -> None:
    stats = trade_statistics(_log([float("nan"), float("nan")]))
    assert stats.n_trades == 0
    assert stats.profit_factor == 0.0


def test_integrates_with_engine_trade_log() -> None:
    # a clean up-then-down price path with a long-then-flat signal
    idx = pd.date_range("2020-01-01", periods=10, freq="B")
    close = pd.Series([100, 101, 102, 103, 104, 103, 102, 101, 100, 99], index=idx, dtype=float)
    signal = pd.Series([0, 1, 1, 1, 0, 0, 0, 0, 0, 0], index=idx)
    _, trade_log = backtest_strategy(pd.DataFrame({"close": close, "signal": signal}))
    stats = trade_statistics(trade_log)
    assert stats.n_trades == len(trade_log)
    assert stats.n_trades >= 1
    assert np.isfinite(stats.avg_return)
