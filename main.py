"""Main entry point for the momentum backtesting pipeline.

Loads data, generates signals, runs a cost-aware backtest with optional
risk management, and exports results.
"""

from pathlib import Path

import matplotlib.pyplot as plt

from src.data.loader import load_yahoo_ohlcv
from src.strategy.momentum import momentum_strategy
from src.backtest.engine import backtest_strategy
from src.reporting.metrics import calculate_metrics
from src.risk.manager import RiskConfig, summarise_risk_events


def main() -> None:
    # --- configuration ---
    ticker = "SPY"
    start_date = "2010-01-01"
    transaction_cost = 0.001

    lookback = 200
    threshold = 0.005
    use_sma_filter = True

    vol_target = 0.15
    vol_window = 20

    risk_config = RiskConfig(
        stop_loss=0.05,
        take_profit=0.10,
        trailing_stop=0.03,
        max_position=1.0,
        daily_loss_limit=0.02,
    )

    # --- pipeline ---
    print("Loading data...")
    df = load_yahoo_ohlcv(ticker=ticker, start=start_date)

    print("Generating signals...")
    strategy_df = momentum_strategy(
        df,
        lookback=lookback,
        threshold=threshold,
        use_sma_filter=use_sma_filter,
    )

    print("Running backtest...")
    backtest_df, trade_log = backtest_strategy(
        strategy_df,
        transaction_cost=transaction_cost,
        vol_target=vol_target,
        vol_window=vol_window,
        risk_config=risk_config,
    )

    # --- metrics ---
    metrics = calculate_metrics(backtest_df["strategy_returns"])
    metrics["Trades (closed)"] = len(trade_log)

    print("\n=== Strategy Metrics ===")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")

    # --- risk events summary ---
    risk_events = summarise_risk_events(backtest_df)
    if risk_events:
        print("\n=== Risk Events ===")
        for event, count in risk_events.items():
            print(f"  {event}: {count}")

    # --- trade log ---
    print("\n=== Trade Log (last 5) ===")
    if len(trade_log) > 0:
        print(trade_log.tail(5).to_string(index=False))
    else:
        print("  No trades found.")

    # --- save outputs ---
    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)

    trade_log.to_csv(output_dir / "trade_log_spy_mom.csv", index=False)
    print(f"\nSaved: {output_dir / 'trade_log_spy_mom.csv'}")

    # equity curve vs buy-and-hold
    backtest_df["buy_hold_returns"] = backtest_df["close"].pct_change().fillna(0)
    backtest_df["buy_hold_equity"] = (1 + backtest_df["buy_hold_returns"]).cumprod()

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(backtest_df.index, backtest_df["equity_curve"], label="Strategy")
    ax.plot(backtest_df.index, backtest_df["buy_hold_equity"], label="Buy & Hold SPY")
    ax.set_title(
        f"{ticker} Momentum Strategy vs Buy-and-Hold\n"
        f"lookback={lookback}, threshold={threshold}, "
        f"SMA200={use_sma_filter}, vol_target={vol_target}"
    )
    ax.set_xlabel("Date")
    ax.set_ylabel("Equity")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()

    equity_plot_file = output_dir / "equity_vs_buyhold.png"
    fig.savefig(equity_plot_file, dpi=150)
    plt.close(fig)
    print(f"Saved: {equity_plot_file}")

    # summary table
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
    existing_cols = [c for c in cols_to_show if c in backtest_df.columns]
    print(backtest_df[existing_cols].tail(10))


if __name__ == "__main__":
    main()
