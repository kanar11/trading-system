import pandas as pd

from src.strategy.momentum import momentum_signal
from src.backtest.engine import SimpleBacktester, BacktestConfig
from src.reporting.metrics import compute_metrics
from src.reporting.trades import build_trade_log


def run_momentum_sweep(
    df: pd.DataFrame,
    lookbacks: list[int],
    thresholds: list[float],
    cfg: BacktestConfig,
) -> pd.DataFrame:
    rows = []

    for lookback in lookbacks:
        for threshold in thresholds:
            sig = momentum_signal(df["close"], lookback=lookback, threshold=threshold)

            bt = SimpleBacktester(df, cfg)
            res = bt.run(sig)

            metrics = compute_metrics(res)
            trades = build_trade_log(res)

            metrics["lookback"] = lookback
            metrics["threshold"] = threshold
            metrics["Trades (closed)"] = len(trades)

            rows.append(metrics)

    out = pd.DataFrame(rows).sort_values(["lookback", "threshold"]).reset_index(drop=True)
    return out