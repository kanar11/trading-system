from pathlib import Path
import pandas as pd
from src.data.loader import load_yahoo_ohlcv
from src.strategy.momentum import momentum_strategy
from src.backtest.engine import backtest_strategy
from src.reporting.metrics import calculate_metrics


def run_sweep(ticker="SPY", start_date="2015-01-01", transaction_cost=0.001):
    lookbacks = [5, 10, 20, 50, 100, 200]
    thresholds = [0.0, 0.005, 0.01, 0.02]

    print("Loading data...")
    df = load_yahoo_ohlcv(ticker=ticker, start=start_date)

    results = []

    for lookback in lookbacks:
        for threshold in thresholds:
            print(f"Testing lookback={lookback}, threshold={threshold}")

            strategy_df = momentum_strategy(
                df.copy(),
                lookback=lookback,
                threshold=threshold,
                use_sma_filter=True
            )

            backtest_df, trade_log = backtest_strategy(
                strategy_df,
                transaction_cost=transaction_cost,
                vol_target=0.15,
                vol_window=20
            )

            metrics = calculate_metrics(backtest_df["strategy_returns"])

            results.append({
                "lookback": lookback,
                "threshold": threshold,
                "total_return": metrics["Total Return"],
                "cagr": metrics["CAGR"],
                "sharpe": metrics["Sharpe Ratio"],
                "max_drawdown": metrics["Max Drawdown"],
                "num_trades": len(trade_log),
            })

    results_df = pd.DataFrame(results).sort_values(
        by=["sharpe", "total_return"], ascending=False
    ).reset_index(drop=True)

    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)
    results_df.to_csv(output_dir / "sweep_results.csv", index=False)

    print("\n=== Top 10 Results ===")
    print(results_df.head(10))


if __name__ == "__main__":
    run_sweep()