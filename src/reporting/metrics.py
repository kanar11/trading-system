"""Performance metrics for strategy evaluation.

Calculates standard quantitative finance statistics from a return series.
"""

import numpy as np
import pandas as pd


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
