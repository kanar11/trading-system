"""Performance metrics for strategy evaluation.

Provides both portfolio-level statistics (from a return series) and
trade-level analytics (from a trade log DataFrame).
"""

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Portfolio-level metrics
# ---------------------------------------------------------------------------


def calculate_metrics(
    strategy_returns: pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> dict[str, float]:
    """Compute core performance statistics from a periodic return series.

    Sharpe and Sortino are computed on excess returns over the risk-free
    rate and annualised by ``periods_per_year``.

    Args:
        strategy_returns: Periodic returns (not cumulative), typically daily.
        risk_free_rate: Annualised risk-free rate used for excess returns.
        periods_per_year: Number of return periods per year (252 for daily).

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
    final_equity = float(equity_curve.iloc[-1])
    total_return = final_equity - 1.0

    n_days = len(strategy_returns)
    years = n_days / periods_per_year
    if years > 0 and final_equity > 0:
        cagr = final_equity ** (1 / years) - 1.0
    elif final_equity <= 0:
        cagr = -1.0  # capital fully wiped out
    else:
        cagr = 0.0

    # per-period risk-free rate for excess-return calculations
    rf_period = risk_free_rate / periods_per_year
    excess = strategy_returns - rf_period
    ann = np.sqrt(periods_per_year)

    mean_excess = float(excess.mean())
    volatility = float(strategy_returns.std())
    sharpe = float(mean_excess / volatility * ann) if volatility > 0 else 0.0

    # Sortino — annualised downside deviation of excess returns below zero
    downside = excess.clip(upper=0.0)
    downside_dev = float(np.sqrt((downside**2).mean()))
    sortino = float(mean_excess / downside_dev * ann) if downside_dev > 0 else 0.0

    # drawdown
    rolling_max = equity_curve.cummax()
    drawdown = equity_curve / rolling_max - 1.0
    max_drawdown = float(drawdown.min())

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
            avg_holding_win = float(trade_log.loc[returns > 0, "holding_days"].mean())
        if n_loss > 0:
            avg_holding_loss = float(trade_log.loc[returns < 0, "holding_days"].mean())

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
