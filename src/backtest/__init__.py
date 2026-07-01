"""Backtesting engine."""

from src.backtest.curves import drawdown_series, equity_curve
from src.backtest.engine import backtest_strategy
from src.backtest.exposure import ExposureSummary, summarize_exposure
from src.backtest.trades import TradeStats, trade_statistics

__all__ = [
    "backtest_strategy",
    "ExposureSummary",
    "summarize_exposure",
    "equity_curve",
    "drawdown_series",
    "TradeStats",
    "trade_statistics",
]
