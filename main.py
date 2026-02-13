from src.data.loader import load_yahoo_ohlcv
from src.strategy.momentum import momentum_signal
from src.backtest.engine import SimpleBacktester, BacktestConfig
from src.reporting.plots import plot_equity
from src.reporting.metrics import compute_metrics
from src.reporting.trades import build_trade_log

def main() -> None:
    df = load_yahoo_ohlcv("SPY", start="2015-01-01")

    sig = momentum_signal(df["close"], lookback=20, threshold=0.0)

    bt = SimpleBacktester(
        df,
        BacktestConfig(fee_bps=1.0, slippage_bps=1.0, initial_cash=10_000.0)
    )

    res = bt.run(sig)

    trade_log = build_trade_log(res)

    metrics = compute_metrics(res)
    metrics["Trades (closed)"] = len(trade_log)

    print("\n=== Trade Log (last 5) ===")
    print(trade_log.tail(5))

    trade_log.to_csv("trade_log_spy_mom.csv", index=False)
    print("Saved: trade_log_spy_mom.csv")

    if len(trade_log) > 0:
        win_rate = (trade_log["trade_return"] > 0).mean()
        avg_trade = trade_log["trade_return"].mean()

        print(f"\nTrades in log: {len(trade_log)}")
        print(f"Win rate: {win_rate:.2%}")
        print(f"Avg trade return: {avg_trade:.4%}")
    else:
        print("No trades found!")


    print ("\n=== Strategy Metrics ===")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"{k}: {v:.4f}")
        else:
            print(f"{k}: {v}")
    print(res[["close", "signal", "pos", "strat_ret", "equity"]].tail(10))
    plot_equity(res, title="SPY Momentum (lookback=20)", save_path="equity_spy_mom.png")

if __name__ == "__main__":
    main()

