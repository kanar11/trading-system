import numpy as np


def calculate_metrics(strategy_returns):
    strategy_returns = strategy_returns.dropna()

    if len(strategy_returns) == 0:
        return {
            "Total Return": 0.0,
            "CAGR": 0.0,
            "Sharpe Ratio": 0.0,
            "Max Drawdown": 0.0,
        }

    equity_curve = (1 + strategy_returns).cumprod()

    total_return = equity_curve.iloc[-1] - 1

    n_days = len(strategy_returns)
    years = n_days / 252 if n_days > 0 else 0
    cagr = (equity_curve.iloc[-1] ** (1 / years) - 1) if years > 0 else 0.0

    volatility = strategy_returns.std()
    sharpe = (strategy_returns.mean() / volatility) * np.sqrt(252) if volatility > 0 else 0.0

    rolling_max = equity_curve.cummax()
    drawdown = equity_curve / rolling_max - 1
    max_drawdown = drawdown.min()

    return {
        "Total Return": total_return,
        "CAGR": cagr,
        "Sharpe Ratio": sharpe,
        "Max Drawdown": max_drawdown,
    }