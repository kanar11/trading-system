"""Round-trip trade statistics from a trade log.

Turns the per-trade log produced by :func:`src.backtest.backtest_strategy`
(one row per closed round-trip, with a ``trade_return`` column) into the
summary every professional tear-sheet reports: hit rate, profit factor,
average win/loss, payoff ratio, best/worst trade, win/loss streaks, and mean
holding period. Return-based metrics (Sharpe, drawdown) describe the *equity
curve*; these describe the *trades* that produced it.

Pure pandas/numpy; the input is never mutated.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import pandas as pd


@dataclass
class TradeStats:
    """Summary statistics over a set of round-trip trades.

    All monetary/return figures are in the units of the ``return_col`` (the
    engine emits fractional returns, e.g. ``0.05`` = +5%). On an empty trade
    log every field is 0.

    Attributes:
        n_trades: Number of round-trip trades.
        n_wins: Trades with a strictly positive return.
        n_losses: Trades with a strictly negative return.
        win_rate: ``n_wins / n_trades`` (0 when there are no trades).
        avg_return: Mean return across all trades (the system's per-trade
            expectancy).
        avg_win: Mean return of winning trades (0 if none).
        avg_loss: Mean return of losing trades, negative or 0.
        best_trade: Largest single-trade return.
        worst_trade: Smallest (most negative) single-trade return.
        gross_profit: Sum of winning-trade returns (>= 0).
        gross_loss: Absolute sum of losing-trade returns (>= 0).
        profit_factor: ``gross_profit / gross_loss``; ``inf`` when there are
            wins but no losses, 0 when there are no wins.
        payoff_ratio: ``avg_win / |avg_loss|``; ``inf`` when there are wins but
            no losses, 0 when there are no wins.
        max_consecutive_wins: Longest run of consecutive winning trades.
        max_consecutive_losses: Longest run of consecutive losing trades.
        avg_holding_days: Mean holding period in days (0 if unavailable).
    """

    n_trades: int
    n_wins: int
    n_losses: int
    win_rate: float
    avg_return: float
    avg_win: float
    avg_loss: float
    best_trade: float
    worst_trade: float
    gross_profit: float
    gross_loss: float
    profit_factor: float
    payoff_ratio: float
    max_consecutive_wins: int
    max_consecutive_losses: int
    avg_holding_days: float


def _max_streak(mask: pd.Series) -> int:
    """Longest run of consecutive ``True`` values in a boolean series."""
    best = current = 0
    for flag in mask:
        current = current + 1 if flag else 0
        best = max(best, current)
    return best


def _safe_ratio(numerator: float, denominator: float) -> float:
    """``numerator / denominator``, or ``inf``/0 when the denominator is 0.

    Returns ``inf`` when the numerator is positive (an unbounded win/loss
    ratio) and 0 when both are 0 (no data).
    """
    if denominator > 0:
        return numerator / denominator
    return math.inf if numerator > 0 else 0.0


def trade_statistics(
    trade_log: pd.DataFrame,
    return_col: str = "trade_return",
    holding_col: str = "holding_days",
) -> TradeStats:
    """Compute round-trip trade statistics from a trade log.

    Args:
        trade_log: One row per closed trade, as returned by
            :func:`src.backtest.backtest_strategy`. Must contain ``return_col``
            (unless empty).
        return_col: Column holding each trade's fractional return.
        holding_col: Optional column holding each trade's holding period in
            days; ignored (and ``avg_holding_days`` set to 0) when absent.

    Returns:
        A populated :class:`TradeStats` (all-zero for an empty log).

    Raises:
        ValueError: If the log is non-empty but lacks ``return_col``.
    """
    if trade_log.empty:
        return TradeStats(0, 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0, 0.0)
    if return_col not in trade_log.columns:
        raise ValueError(f"trade_log must contain a {return_col!r} column.")

    r = trade_log[return_col].astype(float).dropna()
    if r.empty:
        return TradeStats(0, 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0, 0.0)

    wins = r[r > 0]
    losses = r[r < 0]
    n_trades = int(len(r))
    n_wins = int(len(wins))
    n_losses = int(len(losses))

    gross_profit = float(wins.sum())
    gross_loss = float(-losses.sum())  # >= 0
    avg_win = float(wins.mean()) if n_wins else 0.0
    avg_loss = float(losses.mean()) if n_losses else 0.0  # <= 0

    profit_factor = _safe_ratio(gross_profit, gross_loss)
    payoff_ratio = _safe_ratio(avg_win, abs(avg_loss))

    if holding_col in trade_log.columns:
        holding = trade_log[holding_col].astype(float).dropna()
        avg_holding_days = float(holding.mean()) if not holding.empty else 0.0
    else:
        avg_holding_days = 0.0

    return TradeStats(
        n_trades=n_trades,
        n_wins=n_wins,
        n_losses=n_losses,
        win_rate=n_wins / n_trades,
        avg_return=float(r.mean()),
        avg_win=avg_win,
        avg_loss=avg_loss,
        best_trade=float(r.max()),
        worst_trade=float(r.min()),
        gross_profit=gross_profit,
        gross_loss=gross_loss,
        profit_factor=profit_factor,
        payoff_ratio=payoff_ratio,
        max_consecutive_wins=_max_streak(r > 0),
        max_consecutive_losses=_max_streak(r < 0),
        avg_holding_days=avg_holding_days,
    )
