"""Walk-forward validation framework.

Splits historical data into rolling in-sample (training) and out-of-sample
(testing) windows. For each window the strategy is calibrated on in-sample
data and evaluated on out-of-sample data, producing a realistic estimate
of live performance.

Typical usage:
    results = run_walk_forward(df, strategy_fn, metric_fn)
"""

import logging
from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from src.reporting.metrics import calculate_metrics

logger = logging.getLogger(__name__)


@dataclass
class WalkForwardConfig:
    """Configuration for walk-forward validation.

    Attributes:
        in_sample_days: Number of trading days for the in-sample (training) window.
        out_of_sample_days: Number of trading days for the out-of-sample (test) window.
        step_days: Number of days to advance between each fold. Defaults to
            out_of_sample_days (non-overlapping OOS windows).
        min_trades: Minimum trades required in OOS window for a valid fold.
    """

    in_sample_days: int = 504    # ~2 years
    out_of_sample_days: int = 126  # ~6 months
    step_days: int | None = None   # defaults to out_of_sample_days
    min_trades: int = 5

    def __post_init__(self) -> None:
        if self.step_days is None:
            self.step_days = self.out_of_sample_days


@dataclass
class FoldResult:
    """Results from a single walk-forward fold.

    Attributes:
        fold: Fold number (1-indexed).
        is_start: Start date of in-sample window.
        is_end: End date of in-sample window.
        oos_start: Start date of out-of-sample window.
        oos_end: End date of out-of-sample window.
        is_metrics: Performance metrics on in-sample data.
        oos_metrics: Performance metrics on out-of-sample data.
        oos_returns: Daily return series from out-of-sample period.
        num_trades: Number of trades in out-of-sample period.
    """

    fold: int
    is_start: str
    is_end: str
    oos_start: str
    oos_end: str
    is_metrics: dict[str, float]
    oos_metrics: dict[str, float]
    oos_returns: pd.Series
    num_trades: int


def run_walk_forward(
    df: pd.DataFrame,
    strategy_fn: Callable[[pd.DataFrame], pd.DataFrame],
    backtest_fn: Callable[[pd.DataFrame], tuple[pd.DataFrame, pd.DataFrame]],
    config: WalkForwardConfig | None = None,
) -> dict:
    """Run walk-forward validation on a strategy.

    For each rolling window:
    1. Apply strategy_fn to in-sample data (for diagnostics).
    2. Apply strategy_fn to the full window (IS + OOS), then evaluate
       only the OOS portion to avoid look-ahead bias.

    Args:
        df: Full DataFrame with at least 'close' column.
        strategy_fn: Function that takes a DataFrame and returns it with
            a 'signal' column added. Signature: (df) -> df_with_signals.
        backtest_fn: Function that takes a signal DataFrame and returns
            (backtest_df, trade_log_df). Signature: (df) -> (bt_df, trades).
        config: Walk-forward configuration. Uses defaults if None.

    Returns:
        Dictionary with:
            - 'folds': list of FoldResult objects
            - 'summary': aggregated OOS metrics
            - 'oos_equity': combined OOS equity curve
            - 'degradation': IS vs OOS performance comparison
    """
    if config is None:
        config = WalkForwardConfig()

    n = len(df)
    window_size = config.in_sample_days + config.out_of_sample_days

    if n < window_size:
        raise ValueError(
            f"Not enough data: need {window_size} rows but got {n}. "
            f"Reduce in_sample_days ({config.in_sample_days}) or "
            f"out_of_sample_days ({config.out_of_sample_days})."
        )

    folds: list[FoldResult] = []
    all_oos_returns: list[pd.Series] = []
    fold_num = 0

    start = 0
    while start + window_size <= n:
        fold_num += 1
        is_start = start
        is_end = start + config.in_sample_days
        oos_start = is_end
        oos_end = oos_start + config.out_of_sample_days

        # clip to available data
        oos_end = min(oos_end, n)

        is_data = df.iloc[is_start:is_end].copy()
        full_window = df.iloc[is_start:oos_end].copy()

        logger.info(
            "Fold %d: IS [%s → %s] OOS [%s → %s]",
            fold_num,
            df.index[is_start].strftime("%Y-%m-%d"),
            df.index[is_end - 1].strftime("%Y-%m-%d"),
            df.index[oos_start].strftime("%Y-%m-%d"),
            df.index[oos_end - 1].strftime("%Y-%m-%d"),
        )

        # --- in-sample evaluation (for comparison) ---
        is_signals = strategy_fn(is_data)
        is_bt, is_trades = backtest_fn(is_signals)
        is_metrics = calculate_metrics(is_bt["strategy_returns"])

        # --- out-of-sample evaluation ---
        # run strategy on full window to avoid warm-up artefacts at OOS boundary
        full_signals = strategy_fn(full_window)
        full_bt, full_trades = backtest_fn(full_signals)

        # extract only OOS portion
        oos_mask = full_bt.index >= df.index[oos_start]
        oos_returns = full_bt.loc[oos_mask, "strategy_returns"]
        oos_metrics = calculate_metrics(oos_returns)

        # count OOS trades
        oos_trade_count = 0
        if not full_trades.empty and "entry_date" in full_trades.columns:
            oos_trade_count = int(
                (full_trades["entry_date"] >= df.index[oos_start]).sum()
            )

        fold_result = FoldResult(
            fold=fold_num,
            is_start=df.index[is_start].strftime("%Y-%m-%d"),
            is_end=df.index[is_end - 1].strftime("%Y-%m-%d"),
            oos_start=df.index[oos_start].strftime("%Y-%m-%d"),
            oos_end=df.index[min(oos_end - 1, n - 1)].strftime("%Y-%m-%d"),
            is_metrics=is_metrics,
            oos_metrics=oos_metrics,
            oos_returns=oos_returns,
            num_trades=oos_trade_count,
        )
        folds.append(fold_result)
        all_oos_returns.append(oos_returns)

        start += config.step_days

    if not folds:
        raise ValueError("No folds could be created with the given configuration.")

    # --- aggregate results ---
    combined_oos = pd.concat(all_oos_returns)
    combined_oos = combined_oos[~combined_oos.index.duplicated(keep="first")]
    combined_oos = combined_oos.sort_index()

    oos_equity = (1 + combined_oos).cumprod()
    summary_metrics = calculate_metrics(combined_oos)
    summary_metrics["Total Folds"] = len(folds)
    summary_metrics["Avg OOS Sharpe"] = float(
        np.mean([f.oos_metrics["Sharpe Ratio"] for f in folds])
    )
    summary_metrics["Std OOS Sharpe"] = float(
        np.std([f.oos_metrics["Sharpe Ratio"] for f in folds])
    )

    # --- IS vs OOS degradation ---
    avg_is_sharpe = float(np.mean([f.is_metrics["Sharpe Ratio"] for f in folds]))
    avg_oos_sharpe = summary_metrics["Avg OOS Sharpe"]
    degradation = (
        (avg_is_sharpe - avg_oos_sharpe) / abs(avg_is_sharpe)
        if avg_is_sharpe != 0
        else 0.0
    )

    return {
        "folds": folds,
        "summary": summary_metrics,
        "oos_equity": oos_equity,
        "degradation": {
            "avg_is_sharpe": avg_is_sharpe,
            "avg_oos_sharpe": avg_oos_sharpe,
            "sharpe_degradation_pct": degradation * 100,
        },
    }


def print_walk_forward_report(results: dict) -> None:
    """Print a human-readable walk-forward validation report.

    Args:
        results: Dictionary returned by run_walk_forward.
    """
    print("\n" + "=" * 60)
    print("WALK-FORWARD VALIDATION REPORT")
    print("=" * 60)

    # per-fold summary
    print(f"\n{'Fold':>4}  {'OOS Period':<25}  {'Sharpe':>8}  {'Return':>8}  {'MaxDD':>8}  {'Trades':>6}")
    print("-" * 70)
    for f in results["folds"]:
        period = f"{f.oos_start} → {f.oos_end}"
        print(
            f"{f.fold:>4}  {period:<25}  "
            f"{f.oos_metrics['Sharpe Ratio']:>8.2f}  "
            f"{f.oos_metrics['Total Return']:>7.1%}  "
            f"{f.oos_metrics['Max Drawdown']:>7.1%}  "
            f"{f.num_trades:>6}"
        )

    # aggregate
    s = results["summary"]
    print(f"\n{'--- Aggregated OOS Results ---':^70}")
    print(f"  Combined Sharpe:   {s['Sharpe Ratio']:.2f}")
    print(f"  Combined Return:   {s['Total Return']:.1%}")
    print(f"  Combined MaxDD:    {s['Max Drawdown']:.1%}")
    print(f"  Avg OOS Sharpe:    {s['Avg OOS Sharpe']:.2f} (std: {s['Std OOS Sharpe']:.2f})")
    print(f"  Total Folds:       {s['Total Folds']}")

    # degradation
    d = results["degradation"]
    print(f"\n{'--- IS vs OOS Degradation ---':^70}")
    print(f"  Avg IS Sharpe:     {d['avg_is_sharpe']:.2f}")
    print(f"  Avg OOS Sharpe:    {d['avg_oos_sharpe']:.2f}")
    print(f"  Degradation:       {d['sharpe_degradation_pct']:.1f}%")

    if d["sharpe_degradation_pct"] > 50:
        print("  [!] High degradation — strategy may be overfitting in-sample data.")
    elif d["sharpe_degradation_pct"] < 20:
        print("  Strategy shows good robustness across folds.")

    print("=" * 60)
