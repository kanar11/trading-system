"""Performance metrics for strategy evaluation.

Provides both portfolio-level statistics (from a return series) and
trade-level analytics (from a trade log DataFrame).
"""

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Portfolio-level metrics
# ---------------------------------------------------------------------------

def calculate_metrics(strategy_returns: pd.Series) -> dict[str, float]:
    """Compute core performance statistics from a daily return series.

    Args:
        strategy_returns: Daily returns (not cumulative).

    Returns:
        Dictionary with Total Return, CAGR, Sharpe Ratio, Sortino Ratio,
        Max Drawdown, and Calmar Ratio.
    """
    strategy_returns = strategy_returns.dropna()

    if len(strategy_returns) == 0:
        return {
            "Total Return": 0.0,
            "CAGR": 0.0,
            "Sharpe Ratio": 0.0,
            "Sortino Ratio": 0.0,
            "Max Drawdown": 0.0,
            "Calmar Ratio": 0.0,
        }

    equity_curve = (1 + strategy_returns).cumprod()
    total_return = equity_curve.iloc[-1] - 1

    n_days = len(strategy_returns)
    years = n_days / 252 if n_days > 0 else 0
    cagr = (equity_curve.iloc[-1] ** (1 / years) - 1) if years > 0 else 0.0

    volatility = strategy_returns.std()
    sharpe = (
        (strategy_returns.mean() / volatility) * np.sqrt(252)
        if volatility > 0
        else 0.0
    )

    # Sortino — downside deviation only
    downside = strategy_returns[strategy_returns < 0]
    downside_std = downside.std() if len(downside) > 1 else 0.0
    if pd.isna(downside_std):
        downside_std = 0.0
    sortino = (
        (strategy_returns.mean() / downside_std) * np.sqrt(252)
        if downside_std > 0
        else 0.0
    )

    # drawdown
    rolling_max = equity_curve.cummax()
    drawdown = equity_curve / rolling_max - 1
    max_drawdown = drawdown.min()

    # Calmar = CAGR / |Max Drawdown|
    calmar = abs(cagr / max_drawdown) if max_drawdown != 0 else 0.0

    return {
        "Total Return": total_return,
        "CAGR": cagr,
        "Sharpe Ratio": sharpe,
        "Sortino Ratio": sortino,
        "Max Drawdown": max_drawdown,
        "Calmar Ratio": calmar,
    }


# ---------------------------------------------------------------------------
# Trade-level analytics
# ---------------------------------------------------------------------------

def calculate_trade_stats(trade_log: pd.DataFrame) -> dict[str, float | int]:
    """Compute trade-level statistics from a trade log.

    Expects a DataFrame with at least a 'trade_return' column. Optionally
    uses 'holding_days' and 'direction' columns for richer stats.

    Args:
        trade_log: DataFrame with one row per round-trip trade.

    Returns:
        Dictionary with win rate, profit factor, expectancy, avg win/loss,
        streak analysis, and more.
    """
    if trade_log.empty or "trade_return" not in trade_log.columns:
        return _empty_trade_stats()

    returns = trade_log["trade_return"]
    n_trades = len(returns)

    winners = returns[returns > 0]
    losers = returns[returns < 0]
    breakeven = returns[returns == 0]

    n_win = len(winners)
    n_loss = len(losers)

    win_rate = n_win / n_trades if n_trades > 0 else 0.0

    avg_win = float(winners.mean()) if n_win > 0 else 0.0
    avg_loss = float(losers.mean()) if n_loss > 0 else 0.0

    # profit factor = gross wins / gross losses
    gross_win = float(winners.sum()) if n_win > 0 else 0.0
    gross_loss = float(losers.abs().sum()) if n_loss > 0 else 0.0
    profit_factor = gross_win / gross_loss if gross_loss > 0 else float("inf")

    # expectancy = avg return per trade
    expectancy = float(returns.mean())

    # payoff ratio = avg win / |avg loss|
    payoff_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else float("inf")

    # largest win and loss
    largest_win = float(winners.max()) if n_win > 0 else 0.0
    largest_loss = float(losers.min()) if n_loss > 0 else 0.0

    # streak analysis
    max_win_streak, max_loss_streak = _calculate_streaks(returns)

    # holding period stats
    avg_holding = 0.0
    avg_holding_win = 0.0
    avg_holding_loss = 0.0
    if "holding_days" in trade_log.columns:
        avg_holding = float(trade_log["holding_days"].mean())
        if n_win > 0:
            avg_holding_win = float(
                trade_log.loc[returns > 0, "holding_days"].mean()
            )
        if n_loss > 0:
            avg_holding_loss = float(
                trade_log.loc[returns < 0, "holding_days"].mean()
            )

    # direction breakdown
    n_long = 0
    n_short = 0
    if "direction" in trade_log.columns:
        n_long = int((trade_log["direction"] == 1).sum())
        n_short = int((trade_log["direction"] == -1).sum())

    return {
        "Total Trades": n_trades,
        "Winners": n_win,
        "Losers": n_loss,
        "Breakeven": len(breakeven),
        "Win Rate": win_rate,
        "Profit Factor": profit_factor,
        "Payoff Ratio": payoff_ratio,
        "Expectancy": expectancy,
        "Avg Win": avg_win,
        "Avg Loss": avg_loss,
        "Largest Win": largest_win,
        "Largest Loss": largest_loss,
        "Max Win Streak": max_win_streak,
        "Max Loss Streak": max_loss_streak,
        "Avg Holding Days": avg_holding,
        "Avg Holding (Win)": avg_holding_win,
        "Avg Holding (Loss)": avg_holding_loss,
        "Long Trades": n_long,
        "Short Trades": n_short,
    }


def _calculate_streaks(returns: pd.Series) -> tuple[int, int]:
    """Find the longest winning and losing streaks.

    Args:
        returns: Series of trade returns.

    Returns:
        Tuple of (max_win_streak, max_loss_streak).
    """
    max_win = 0
    max_loss = 0
    current_win = 0
    current_loss = 0

    for r in returns:
        if r > 0:
            current_win += 1
            current_loss = 0
            max_win = max(max_win, current_win)
        elif r < 0:
            current_loss += 1
            current_win = 0
            max_loss = max(max_loss, current_loss)
        else:
            current_win = 0
            current_loss = 0

    return max_win, max_loss


def _empty_trade_stats() -> dict[str, float | int]:
    """Return zeroed trade stats for empty trade logs."""
    return {
        "Total Trades": 0,
        "Winners": 0,
        "Losers": 0,
        "Breakeven": 0,
        "Win Rate": 0.0,
        "Profit Factor": 0.0,
        "Payoff Ratio": 0.0,
        "Expectancy": 0.0,
        "Avg Win": 0.0,
        "Avg Loss": 0.0,
        "Largest Win": 0.0,
        "Largest Loss": 0.0,
        "Max Win Streak": 0,
        "Max Loss Streak": 0,
        "Avg Holding Days": 0.0,
        "Avg Holding (Win)": 0.0,
        "Avg Holding (Loss)": 0.0,
        "Long Trades": 0,
        "Short Trades": 0,
    }
