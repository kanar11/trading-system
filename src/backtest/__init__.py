"""Backtesting engine."""

from src.backtest.curves import drawdown_series, equity_curve
from src.backtest.engine import backtest_strategy
from src.backtest.excursions import trade_excursions
from src.backtest.exposure import ExposureSummary, summarize_exposure
from src.backtest.robustness import lag_sensitivity
from src.backtest.signal_bridge import SignalFollowStrategy, run_signal_event_backtest
from src.backtest.trades import TradeStats, trade_statistics
from src.backtest.walk_forward_weights import (
    WalkForwardWeightsResult,
    walk_forward_weights,
)
from src.backtest.weights import backtest_weights

__all__ = [
    "backtest_strategy",
    "ExposureSummary",
    "summarize_exposure",
    "equity_curve",
    "drawdown_series",
    "TradeStats",
    "trade_statistics",
    "backtest_weights",
    "lag_sensitivity",
    "WalkForwardWeightsResult",
    "walk_forward_weights",
    "SignalFollowStrategy",
    "run_signal_event_backtest",
    "trade_excursions",
]
