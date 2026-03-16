from pathlib import Path

import matplotlib.pyplot as plt

from src.data.loader import load_yahoo_ohlcv
from src.strategy.momentum import momentum_strategy
from src.backtest.engine import backtest_strategy
from src.reporting.metrics import calculate_metrics


def main() -> None:
    ticker = "SPY"
    start_date = "2010-01-01"
    transaction_cost = 0.001

    lookback = 200
    threshold = 0.005
    use_sma_filter = True

    vol_target = 0.15
    vol_window = 20

    print("Loading data...")
    df = load_yahoo_ohlcv(ticker=ticker, start=start_date)

    print("Generating signals...")
    strategy_df = momentum_strategy(
        df,
        lookback=lookback,
        threshold=threshold,
        use_sma_filter=use_sma_filter
    )

    print("Running backtest...")
    backtest_df, trade_log = backtest_strategy(
        strategy_df,
        transaction_cost=transaction_cost,
        vol_target=vol_target,
        vol_window=vol_window
    )

    metrics = calculate_metrics(backtest_df["strategy_returns"])
    metrics["Trades (closed)"] = len(trade_log)

    print("\n=== Trade Log (last 5) ===")
    if len(trade_log) > 0:
        print(trade_log.tail(5))
    else:
        print("No trades found.")

    print("\n=== Strategy Metrics ===")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"{k}: {v:.4f}")
        else:
            print(f"{k}: {v}")

    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)

    trade_log_file = output_dir / "trade_log_spy_mom.csv"
    trade_log.to_csv(trade_log_file, index=False)
    print(f"\nSaved: {trade_log_file}")

    backtest_df["buy_hold_returns"] = backtest_df["close"].pct_change().fillna(0)
    backtest_df["buy_hold_equity"] = (1 + backtest_df["buy_hold_returns"]).cumprod()

    equity_plot_file = output_dir / "equity_vs_buyhold.png"
    plt.figure(figsize=(10, 6))
    plt.plot(backtest_df.index, backtest_df["equity_curve"], label="Strategy")
    plt.plot(backtest_df.index, backtest_df["buy_hold_equity"], label="Buy & Hold SPY")
    plt.title(
        f"{ticker} Momentum Strategy vs Buy-and-Hold\n"
        f"lookback={lookback}, threshold={threshold}, "
        f"SMA200={use_sma_filter}, vol_target={vol_target}"
    )
    plt.xlabel("Date")
    plt.ylabel("Equity")
    plt.legend()
    plt.tight_layout()
    plt.savefig(equity_plot_file)
    plt.close()
    print(f"Saved: {equity_plot_file}")

    print("\n=== Backtest Data (last 10 rows) ===")
    cols_to_show = [
        "close",
        "signal",
        "position",
        "scaled_position",
        "strategy_returns",
        "equity_curve",
        "buy_hold_equity",
    ]
    existing_cols = [col for col in cols_to_show if col in backtest_df.columns]
    print(backtest_df[existing_cols].tail(10))


if __name__ == "__main__":
    main()