"""Parameter sweep runner for strategy optimisation.

Evaluates a momentum strategy across a grid of lookback / threshold
combinations and exports ranked results to CSV.
"""

import argparse
import logging
from pathlib import Path

import pandas as pd

from src.data.loader import load_yahoo_ohlcv
from src.strategy.momentum import momentum_strategy
from src.backtest.engine import backtest_strategy
from src.reporting.metrics import calculate_metrics

logger = logging.getLogger(__name__)


def run_sweep(
    ticker: str = "SPY",
    start_date: str = "2015-01-01",
    transaction_cost: float = 0.001,
    lookbacks: list[int] | None = None,
    thresholds: list[float] | None = None,
    output_dir: str = "results",
) -> pd.DataFrame:
    """Run a parameter sweep and return ranked results.

    Args:
        ticker: Instrument symbol.
        start_date: Data start date.
        transaction_cost: Cost per trade as a fraction.
        lookbacks: List of lookback periods to test.
        thresholds: List of momentum thresholds to test.
        output_dir: Directory to save CSV output.

    Returns:
        DataFrame of results sorted by Sharpe Ratio descending.
    """
    if lookbacks is None:
        lookbacks = [5, 10, 20, 50, 100, 200]
    if thresholds is None:
        thresholds = [0.0, 0.005, 0.01, 0.02]

    logger.info("Loading %s data from %s...", ticker, start_date)
    df = load_yahoo_ohlcv(ticker=ticker, start=start_date)

    total = len(lookbacks) * len(thresholds)
    results: list[dict] = []

    for i, lookback in enumerate(lookbacks):
        for j, threshold in enumerate(thresholds):
            n = i * len(thresholds) + j + 1
            logger.info("[%d/%d] lookback=%d, threshold=%.4f", n, total, lookback, threshold)

            strategy_df = momentum_strategy(
                df.copy(),
                lookback=lookback,
                threshold=threshold,
                use_sma_filter=True,
            )

            backtest_df, trade_log = backtest_strategy(
                strategy_df,
                transaction_cost=transaction_cost,
                vol_target=0.15,
                vol_window=20,
            )

            metrics = calculate_metrics(backtest_df["strategy_returns"])

            results.append(
                {
                    "ticker": ticker,
                    "lookback": lookback,
                    "threshold": threshold,
                    "total_return": metrics["Total Return"],
                    "cagr": metrics["CAGR"],
                    "sharpe": metrics["Sharpe Ratio"],
                    "sortino": metrics["Sortino Ratio"],
                    "max_drawdown": metrics["Max Drawdown"],
                    "calmar": metrics["Calmar Ratio"],
                    "num_trades": len(trade_log),
                }
            )

    results_df = (
        pd.DataFrame(results)
        .sort_values(by=["sharpe", "total_return"], ascending=False)
        .reset_index(drop=True)
    )

    out = Path(output_dir)
    out.mkdir(exist_ok=True)
    output_file = out / "sweep_results.csv"
    results_df.to_csv(output_file, index=False)

    logger.info("Saved: %s", output_file)

    print("\n=== Top 10 Results ===")
    print(results_df.head(10).to_string(index=False))

    return results_df


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        level=logging.INFO,
    )
    run_sweep()
