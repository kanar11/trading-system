"""Tests for performance metrics and trade-level analytics."""

import pandas as pd
import pytest

from src.reporting.metrics import calculate_metrics, calculate_trade_stats


# ---------------------------------------------------------------------------
# Portfolio-level metrics
# ---------------------------------------------------------------------------

class TestCalculateMetrics:
    def test_positive_returns(self):
        returns = pd.Series([0.01, 0.02, -0.005, 0.015, 0.01])
        metrics = calculate_metrics(returns)

        assert metrics["Total Return"] > 0
        assert metrics["Sharpe Ratio"] > 0
        assert metrics["Max Drawdown"] <= 0

    def test_empty_returns(self):
        metrics = calculate_metrics(pd.Series([], dtype=float))

        assert metrics["Total Return"] == 0.0
        assert metrics["Sharpe Ratio"] == 0.0
        assert metrics["Calmar Ratio"] == 0.0

    def test_all_negative(self):
        metrics = calculate_metrics(pd.Series([-0.01, -0.02, -0.015]))

        assert metrics["Total Return"] < 0
        assert metrics["Max Drawdown"] < 0
        assert metrics["Sharpe Ratio"] < 0

    def test_keys(self):
        metrics = calculate_metrics(pd.Series([0.01, -0.005]))

        expected = {
            "Total Return", "CAGR", "Sharpe Ratio",
            "Sortino Ratio", "Max Drawdown", "Calmar Ratio",
        }
        assert set(metrics.keys()) == expected


# ---------------------------------------------------------------------------
# Trade-level analytics
# ---------------------------------------------------------------------------

def _make_trade_log(returns, directions=None, holding_days=None):
    """Helper to build a trade log DataFrame."""
    data = {"trade_return": returns}
    if directions is not None:
        data["direction"] = directions
    if holding_days is not None:
        data["holding_days"] = holding_days
    return pd.DataFrame(data)


class TestTradeStats:
    def test_basic_win_rate(self):
        log = _make_trade_log([0.05, -0.02, 0.03, -0.01, 0.04])
        stats = calculate_trade_stats(log)

        assert stats["Total Trades"] == 5
        assert stats["Winners"] == 3
        assert stats["Losers"] == 2
        assert stats["Win Rate"] == pytest.approx(0.6)

    def test_profit_factor(self):
        log = _make_trade_log([0.10, -0.05, 0.08])
        stats = calculate_trade_stats(log)

        # gross win = 0.18, gross loss = 0.05
        assert stats["Profit Factor"] == pytest.approx(0.18 / 0.05, rel=1e-6)

    def test_expectancy(self):
        returns = [0.05, -0.02, 0.03]
        log = _make_trade_log(returns)
        stats = calculate_trade_stats(log)

        assert stats["Expectancy"] == pytest.approx(sum(returns) / len(returns))

    def test_payoff_ratio(self):
        log = _make_trade_log([0.10, 0.06, -0.04, -0.02])
        stats = calculate_trade_stats(log)

        avg_win = (0.10 + 0.06) / 2
        avg_loss = (-0.04 + -0.02) / 2
        assert stats["Payoff Ratio"] == pytest.approx(abs(avg_win / avg_loss))

    def test_streaks(self):
        log = _make_trade_log([0.01, 0.02, 0.03, -0.01, -0.02, 0.01])
        stats = calculate_trade_stats(log)

        assert stats["Max Win Streak"] == 3
        assert stats["Max Loss Streak"] == 2

    def test_largest_win_loss(self):
        log = _make_trade_log([0.01, 0.15, -0.03, -0.08, 0.05])
        stats = calculate_trade_stats(log)

        assert stats["Largest Win"] == pytest.approx(0.15)
        assert stats["Largest Loss"] == pytest.approx(-0.08)

    def test_holding_days(self):
        log = _make_trade_log(
            [0.05, -0.02, 0.03],
            holding_days=[10, 5, 15],
        )
        stats = calculate_trade_stats(log)

        assert stats["Avg Holding Days"] == pytest.approx(10.0)
        assert stats["Avg Holding (Win)"] == pytest.approx(12.5)
        assert stats["Avg Holding (Loss)"] == pytest.approx(5.0)

    def test_direction_breakdown(self):
        log = _make_trade_log([0.05, -0.02, 0.03], directions=[1, -1, 1])
        stats = calculate_trade_stats(log)

        assert stats["Long Trades"] == 2
        assert stats["Short Trades"] == 1

    def test_empty_trade_log(self):
        stats = calculate_trade_stats(pd.DataFrame())

        assert stats["Total Trades"] == 0
        assert stats["Win Rate"] == 0.0
        assert stats["Profit Factor"] == 0.0

    def test_all_winners(self):
        log = _make_trade_log([0.05, 0.03, 0.02])
        stats = calculate_trade_stats(log)

        assert stats["Win Rate"] == 1.0
        assert stats["Profit Factor"] == float("inf")
        assert stats["Losers"] == 0

    def test_all_losers(self):
        log = _make_trade_log([-0.05, -0.03, -0.02])
        stats = calculate_trade_stats(log)

        assert stats["Win Rate"] == 0.0
        assert stats["Winners"] == 0
        assert stats["Expectancy"] < 0
