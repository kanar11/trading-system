"""Tests for the risk management module."""

import numpy as np
import pandas as pd
import pytest

from src.risk.manager import RiskConfig, apply_risk_controls, summarise_risk_events


def _make_backtest_df(prices: list[float], positions: list[float]) -> pd.DataFrame:
    """Helper to build a minimal backtest DataFrame for risk tests."""
    dates = pd.date_range("2020-01-01", periods=len(prices), freq="B")
    return pd.DataFrame(
        {
            "close": prices,
            "position": positions,
            "scaled_position": positions,
        },
        index=dates,
    )


class TestStopLoss:
    def test_stop_loss_triggers_on_long(self):
        # price drops 6% from entry — should trigger 5% stop
        prices = [100, 100, 94, 93, 95]
        positions = [0, 1, 1, 1, 1]
        df = _make_backtest_df(prices, positions)

        config = RiskConfig(stop_loss=0.05, take_profit=None, trailing_stop=None, daily_loss_limit=None)
        result = apply_risk_controls(df, config)

        # position should be zeroed at index 2 (6% drop)
        assert result["scaled_position"].iloc[2] == 0.0
        assert result["risk_event"].iloc[2] == "stop_loss"

    def test_no_stop_loss_when_disabled(self):
        prices = [100, 100, 90, 85]
        positions = [0, 1, 1, 1]
        df = _make_backtest_df(prices, positions)

        config = RiskConfig(stop_loss=None, take_profit=None, trailing_stop=None, daily_loss_limit=None)
        result = apply_risk_controls(df, config)

        # all positions should remain
        assert result["scaled_position"].iloc[2] == 1.0
        assert result["scaled_position"].iloc[3] == 1.0


class TestTakeProfit:
    def test_take_profit_triggers(self):
        # price rises 11% from entry — should trigger 10% TP
        prices = [100, 100, 111, 115]
        positions = [0, 1, 1, 1]
        df = _make_backtest_df(prices, positions)

        config = RiskConfig(stop_loss=None, take_profit=0.10, trailing_stop=None, daily_loss_limit=None)
        result = apply_risk_controls(df, config)

        assert result["scaled_position"].iloc[2] == 0.0
        assert result["risk_event"].iloc[2] == "take_profit"


class TestTrailingStop:
    def test_trailing_stop_triggers(self):
        # price goes up then drops 4% from peak — should trigger 3% trailing
        prices = [100, 100, 110, 115, 111]
        positions = [0, 1, 1, 1, 1]
        df = _make_backtest_df(prices, positions)

        config = RiskConfig(stop_loss=None, take_profit=None, trailing_stop=0.03, daily_loss_limit=None)
        result = apply_risk_controls(df, config)

        # peak is 115, drop to 111 is ~3.5% drawdown
        assert result["scaled_position"].iloc[4] == 0.0
        assert result["risk_event"].iloc[4] == "trailing_stop"


class TestMaxPosition:
    def test_position_capped(self):
        prices = [100, 100, 102]
        positions = [0, 2.5, 2.5]
        df = _make_backtest_df(prices, positions)

        config = RiskConfig(
            stop_loss=None, take_profit=None, trailing_stop=None,
            max_position=1.5, daily_loss_limit=None,
        )
        result = apply_risk_controls(df, config)

        assert result["scaled_position"].iloc[1] == 1.5
        assert result["risk_event"].iloc[1] == "pos_capped"


class TestSummariseRiskEvents:
    def test_counts_events(self):
        prices = [100, 100, 94, 93, 95]
        positions = [0, 1, 1, 1, 1]
        df = _make_backtest_df(prices, positions)

        config = RiskConfig(stop_loss=0.05, take_profit=None, trailing_stop=None, daily_loss_limit=None)
        result = apply_risk_controls(df, config)
        summary = summarise_risk_events(result)

        assert "stop_loss" in summary
        assert summary["stop_loss"] >= 1

    def test_empty_when_no_events(self):
        prices = [100, 101, 102]
        positions = [0, 1, 1]
        df = _make_backtest_df(prices, positions)

        config = RiskConfig(stop_loss=None, take_profit=None, trailing_stop=None, daily_loss_limit=None)
        result = apply_risk_controls(df, config)
        summary = summarise_risk_events(result)

        assert summary == {}
