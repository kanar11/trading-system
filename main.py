from src.data.loader import load_yahoo_ohlcv
from src.strategy.momentum import momentum_signal
from src.backtest.engine import SimpleBacktester, BacktestConfig
from src.reporting.plots import plot_equity
from src.reporting.metrics import compute_metrics
from src.reporting.trades import build_trade_log
from src.reporting.sweep import run_momentum_sweep

def main() -> None:
    df = load_yahoo_ohlcv("SPY", start="2010-01-01")
    split_date = "2019-01-01"

    train = df[df.index < split_date]
    test = df[df.index >= split_date]

    sig = momentum_signal(test["close"], lookback=200, threshold=0.02)

    bt = SimpleBacktester(
        test,
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
    print("\n=== Momentum Parameter Sweep ===")

    cfg = BacktestConfig(fee_bps=1.0, slippage_bps=1.0, initial_cash=10_000.0)
    lookbacks = [5, 10, 20, 50, 100, 200]

    thresholds = [0.005, 0.01, 0.02]

    sweep = run_momentum_sweep(
        train,
        lookbacks=lookbacks,
        thresholds=thresholds,
        cfg=cfg
    )

    cols = ["lookback", "threshold", "Total Return", "CAGR", "Sharpe", "Max Drawdown", "Trades", "Trades (closed)"]

    top = sweep[cols].sort_values(["Sharpe"], ascending=False).round(4)

    print("\n=== Top 5 Parameter Sets ===")
    print(top.head(5))

    sweep.to_csv("sweep_results.csv")
    print("Saved: sweep_results.csv")
    plot_equity(res, title="SPY Momentum (lookback=20)", save_path="equity_spy_mom.png")

if __name__ == "__main__":
    main()

