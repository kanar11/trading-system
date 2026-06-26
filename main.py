"""Main entry point for the backtesting pipeline.

Supports momentum, mean reversion, and adaptive (regime-based) strategies
with optional risk management and walk-forward validation.

Usage:
    python main.py
    python main.py --strategy mean-reversion --ticker AAPL
    python main.py --strategy adaptive
    python main.py --lookback 100 --threshold 0.01 --no-risk
    python main.py --walk-forward
"""

import argparse
import logging
from collections.abc import Callable
from functools import partial
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.backtest.engine import backtest_strategy
from src.data.csv_loader import load_csv_ohlcv
from src.data.loader import load_yahoo_ohlcv
from src.execution.slippage import ExecutionConfig, apply_execution_costs
from src.regime.detector import RegimeConfig, adaptive_strategy
from src.reporting.metrics import calculate_metrics, calculate_trade_stats
from src.reporting.tearsheet import generate_tearsheet
from src.risk.manager import RiskConfig, summarise_risk_events
from src.strategy.breakout import breakout_strategy
from src.strategy.ema_crossover import ema_crossover_strategy, macd_strategy
from src.strategy.mean_reversion import mean_reversion_strategy
from src.strategy.momentum import momentum_strategy
from src.validation.monte_carlo import bootstrap_returns, print_monte_carlo_report
from src.validation.stat_tests import (
    deflated_sharpe_ratio,
    probabilistic_sharpe_ratio,
    sharpe_ttest,
)
from src.validation.walk_forward import (
    WalkForwardConfig,
    print_walk_forward_report,
    run_walk_forward,
)

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Systematic strategy backtester with risk management.",
    )

    # data
    parser.add_argument("--ticker", default="SPY", help="Yahoo Finance symbol (default: SPY)")
    parser.add_argument(
        "--start", default="2010-01-01", help="Start date YYYY-MM-DD (default: 2010-01-01)"
    )
    parser.add_argument(
        "--csv",
        default=None,
        help="Load OHLCV from a local CSV instead of Yahoo Finance.",
    )

    # strategy selection
    parser.add_argument(
        "--strategy",
        choices=["momentum", "mean-reversion", "adaptive", "breakout", "ema-cross", "macd"],
        default="momentum",
        help="Strategy type (default: momentum)",
    )

    # momentum params
    parser.add_argument(
        "--lookback", type=int, default=200, help="Momentum lookback period (default: 200)"
    )
    parser.add_argument(
        "--threshold", type=float, default=0.005, help="Signal threshold (default: 0.005)"
    )
    parser.add_argument(
        "--no-sma-filter", action="store_true", help="Disable SMA-200 regime filter"
    )

    # mean reversion params
    parser.add_argument(
        "--bb-window", type=int, default=20, help="Bollinger Bands window (default: 20)"
    )
    parser.add_argument(
        "--bb-std", type=float, default=2.0, help="Bollinger Bands std dev (default: 2.0)"
    )
    parser.add_argument("--rsi-period", type=int, default=14, help="RSI period (default: 14)")
    parser.add_argument(
        "--rsi-oversold", type=float, default=30.0, help="RSI oversold level (default: 30)"
    )
    parser.add_argument(
        "--rsi-overbought", type=float, default=70.0, help="RSI overbought level (default: 70)"
    )
    parser.add_argument(
        "--no-rsi-filter", action="store_true", help="Disable RSI filter for mean reversion"
    )

    # regime detection (adaptive mode)
    parser.add_argument(
        "--adx-period", type=int, default=14, help="ADX period for regime detection (default: 14)"
    )
    parser.add_argument(
        "--adx-threshold", type=float, default=25.0, help="ADX trending threshold (default: 25)"
    )
    parser.add_argument(
        "--hurst-window", type=int, default=100, help="Hurst exponent window (default: 100)"
    )

    # breakout params
    parser.add_argument(
        "--bo-entry", type=int, default=20, help="Donchian entry window (default: 20)"
    )
    parser.add_argument(
        "--bo-exit", type=int, default=10, help="Donchian exit window (default: 10)"
    )
    parser.add_argument(
        "--bo-atr-filter",
        type=float,
        default=0.0,
        help="Breakout size in ATR units (default: 0 = off)",
    )

    # EMA / MACD params
    parser.add_argument("--ema-fast", type=int, default=20, help="Fast EMA span (default: 20)")
    parser.add_argument("--ema-slow", type=int, default=50, help="Slow EMA span (default: 50)")
    parser.add_argument(
        "--ema-gap-bps", type=float, default=0.0, help="EMA spread gap in bps (default: 0)"
    )
    parser.add_argument("--macd-fast", type=int, default=12, help="MACD fast span (default: 12)")
    parser.add_argument("--macd-slow", type=int, default=26, help="MACD slow span (default: 26)")
    parser.add_argument("--macd-signal", type=int, default=9, help="MACD signal span (default: 9)")

    # vol targeting
    parser.add_argument(
        "--vol-target", type=float, default=0.15, help="Annualised vol target (default: 0.15)"
    )
    parser.add_argument(
        "--vol-window", type=int, default=20, help="Realised vol window (default: 20)"
    )

    # risk management
    parser.add_argument("--no-risk", action="store_true", help="Disable all risk controls")
    parser.add_argument(
        "--stop-loss", type=float, default=0.05, help="Stop-loss threshold (default: 0.05)"
    )
    parser.add_argument(
        "--take-profit", type=float, default=0.10, help="Take-profit threshold (default: 0.10)"
    )
    parser.add_argument(
        "--trailing-stop", type=float, default=0.03, help="Trailing stop threshold (default: 0.03)"
    )
    parser.add_argument(
        "--max-position", type=float, default=1.0, help="Max position size (default: 1.0)"
    )
    parser.add_argument(
        "--daily-loss-limit", type=float, default=0.02, help="Daily loss limit (default: 0.02)"
    )

    # walk-forward validation
    parser.add_argument("--walk-forward", action="store_true", help="Run walk-forward validation")
    parser.add_argument(
        "--wf-is-days", type=int, default=504, help="Walk-forward in-sample days (default: 504)"
    )
    parser.add_argument(
        "--wf-oos-days",
        type=int,
        default=126,
        help="Walk-forward out-of-sample days (default: 126)",
    )

    # costs
    parser.add_argument(
        "--cost", type=float, default=0.001, help="Transaction cost (default: 0.001)"
    )
    parser.add_argument(
        "--execution-model",
        action="store_true",
        help="Replace flat --cost with realistic spread + sqrt-impact model.",
    )
    parser.add_argument(
        "--spread-bps", type=float, default=5.0, help="Bid-ask spread bps (default: 5)"
    )
    parser.add_argument(
        "--impact-coeff", type=float, default=0.1, help="Market impact coefficient (default: 0.1)"
    )
    parser.add_argument(
        "--impact-exponent",
        type=float,
        default=0.5,
        help="Market impact exponent (default: 0.5 = sqrt)",
    )

    # robustness
    parser.add_argument(
        "--monte-carlo",
        type=int,
        default=0,
        help="Run N bootstrap MC simulations on the return series.",
    )
    parser.add_argument(
        "--mc-block-size",
        type=int,
        default=1,
        help="Bootstrap block size (1 = i.i.d., >1 = moving block).",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=1,
        help="For Deflated Sharpe Ratio — number of parameter trials run.",
    )

    # output
    parser.add_argument(
        "--output-dir", default="results", help="Output directory (default: results)"
    )
    parser.add_argument(
        "--tearsheet", action="store_true", help="Generate a multi-panel PNG tearsheet."
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable debug logging")

    return parser.parse_args()


def setup_logging(verbose: bool = False) -> None:
    """Configure logging format and level."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        level=level,
    )


def _build_strategy_fn(args: argparse.Namespace) -> Callable[[pd.DataFrame], pd.DataFrame]:
    """Return a strategy function based on CLI args."""
    mom_fn = partial(
        momentum_strategy,
        lookback=args.lookback,
        threshold=args.threshold,
        use_sma_filter=not args.no_sma_filter,
    )
    mr_fn = partial(
        mean_reversion_strategy,
        bb_window=args.bb_window,
        bb_std=args.bb_std,
        rsi_period=args.rsi_period,
        rsi_oversold=args.rsi_oversold,
        rsi_overbought=args.rsi_overbought,
        use_rsi_filter=not args.no_rsi_filter,
    )

    if args.strategy == "adaptive":
        regime_config = RegimeConfig(
            adx_period=args.adx_period,
            adx_trending_threshold=args.adx_threshold,
            hurst_window=args.hurst_window,
        )
        return partial(
            adaptive_strategy,
            momentum_fn=mom_fn,
            mean_reversion_fn=mr_fn,
            config=regime_config,
        )
    if args.strategy == "mean-reversion":
        return mr_fn
    if args.strategy == "breakout":
        return partial(
            breakout_strategy,
            entry_window=args.bo_entry,
            exit_window=args.bo_exit,
            atr_filter=args.bo_atr_filter,
        )
    if args.strategy == "ema-cross":
        return partial(
            ema_crossover_strategy,
            fast=args.ema_fast,
            slow=args.ema_slow,
            gap_bps=args.ema_gap_bps,
        )
    if args.strategy == "macd":
        return partial(
            macd_strategy,
            fast=args.macd_fast,
            slow=args.macd_slow,
            signal_span=args.macd_signal,
        )
    return mom_fn


def _build_backtest_fn(
    args: argparse.Namespace, risk_config: RiskConfig | None
) -> Callable[[pd.DataFrame], tuple[pd.DataFrame, pd.DataFrame]]:
    """Return a backtest function based on CLI args."""
    return partial(
        backtest_strategy,
        transaction_cost=args.cost,
        vol_target=args.vol_target,
        vol_window=args.vol_window,
        risk_config=risk_config,
    )


def main() -> None:
    args = parse_args()
    setup_logging(args.verbose)

    # --- risk config ---
    risk_config = None
    if not args.no_risk:
        risk_config = RiskConfig(
            stop_loss=args.stop_loss,
            take_profit=args.take_profit,
            trailing_stop=args.trailing_stop,
            max_position=args.max_position,
            daily_loss_limit=args.daily_loss_limit,
        )

    # --- load data ---
    if args.csv is not None:
        logger.info("Loading OHLCV from CSV: %s", args.csv)
        df = load_csv_ohlcv(args.csv, start=args.start)
    else:
        logger.info("Loading %s data from %s...", args.ticker, args.start)
        df = load_yahoo_ohlcv(ticker=args.ticker, start=args.start)
    logger.info("Loaded %d rows.", len(df))

    strategy_fn = _build_strategy_fn(args)
    backtest_fn = _build_backtest_fn(args, risk_config)
    strategy_label = args.strategy.replace("-", " ").title()

    # --- walk-forward validation mode ---
    if args.walk_forward:
        logger.info("Running walk-forward validation (%s)...", strategy_label)
        wf_config = WalkForwardConfig(
            in_sample_days=args.wf_is_days,
            out_of_sample_days=args.wf_oos_days,
        )
        results = run_walk_forward(df, strategy_fn, backtest_fn, wf_config)
        print_walk_forward_report(results)

        # save OOS equity curve
        output_dir = Path(args.output_dir)
        output_dir.mkdir(exist_ok=True)

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(results["oos_equity"].index, results["oos_equity"].values, linewidth=1.2)
        ax.set_title(f"{args.ticker} {strategy_label} — Walk-Forward OOS Equity")
        ax.set_xlabel("Date")
        ax.set_ylabel("Equity")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        wf_plot = output_dir / f"walk_forward_oos_{args.ticker.lower()}.png"
        fig.savefig(wf_plot, dpi=150)
        plt.close(fig)
        logger.info("Saved: %s", wf_plot)
        return

    # --- standard backtest mode ---
    logger.info("Generating %s signals...", strategy_label)
    strategy_df = strategy_fn(df)

    logger.info("Running backtest...")
    backtest_df, trade_log = backtest_fn(strategy_df)

    # --- optional realistic execution model ---
    if args.execution_model:
        logger.info("Applying realistic execution-cost model.")
        exec_cfg = ExecutionConfig(
            spread_bps=args.spread_bps,
            impact_coeff=args.impact_coeff,
            impact_exponent=args.impact_exponent,
        )
        backtest_df = apply_execution_costs(backtest_df, exec_cfg)

    # --- metrics ---
    metrics = calculate_metrics(backtest_df["strategy_returns"])
    metrics["Trades (closed)"] = len(trade_log)

    print(f"\n=== {strategy_label} Strategy Metrics ===")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")

    # --- risk events summary ---
    if risk_config is not None:
        risk_events = summarise_risk_events(backtest_df)
        if risk_events:
            print("\n=== Risk Events ===")
            for event, count in risk_events.items():
                print(f"  {event}: {count}")

    # --- trade-level analytics ---
    trade_stats = calculate_trade_stats(trade_log)
    if trade_stats["Total Trades"] > 0:
        print("\n=== Trade Analytics ===")
        print(
            f"  Win Rate:        {trade_stats['Win Rate']:.1%}  "
            f"({trade_stats['Winners']}W / {trade_stats['Losers']}L)"
        )
        print(f"  Profit Factor:   {trade_stats['Profit Factor']:.2f}")
        print(f"  Expectancy:      {trade_stats['Expectancy']:.4f}")
        print(f"  Payoff Ratio:    {trade_stats['Payoff Ratio']:.2f}")
        print(f"  Avg Win:         {trade_stats['Avg Win']:.4f}")
        print(f"  Avg Loss:        {trade_stats['Avg Loss']:.4f}")
        print(f"  Largest Win:     {trade_stats['Largest Win']:.4f}")
        print(f"  Largest Loss:    {trade_stats['Largest Loss']:.4f}")
        print(f"  Max Win Streak:  {trade_stats['Max Win Streak']}")
        print(f"  Max Loss Streak: {trade_stats['Max Loss Streak']}")
        if trade_stats["Avg Holding Days"] > 0:
            print(f"  Avg Holding:     {trade_stats['Avg Holding Days']:.0f} days")
        if trade_stats["Long Trades"] + trade_stats["Short Trades"] > 0:
            print(
                f"  Long/Short:      {trade_stats['Long Trades']}L / {trade_stats['Short Trades']}S"
            )

    # --- trade log ---
    print("\n=== Trade Log (last 5) ===")
    if len(trade_log) > 0:
        print(trade_log.tail(5).to_string(index=False))
    else:
        print("  No trades found.")

    # --- statistical significance ---
    sr_test = sharpe_ttest(backtest_df["strategy_returns"])
    psr = probabilistic_sharpe_ratio(backtest_df["strategy_returns"], target_sharpe=0.0)
    dsr = deflated_sharpe_ratio(backtest_df["strategy_returns"], n_trials=args.n_trials)
    print("\n=== Sharpe Significance ===")
    print(f"  Annualised Sharpe:    {sr_test.sharpe_annualised:>+7.2f}")
    print(f"  t-stat:               {sr_test.t_stat:>+7.2f}")
    print(f"  p-value (two-sided):  {sr_test.p_value_two_sided:>7.4f}")
    print(f"  P(SR > 0) [PSR]:      {psr:>7.3f}")
    print(f"  Deflated SR (trials={args.n_trials}):  {dsr:>7.3f}")

    # --- Monte Carlo ---
    if args.monte_carlo > 0:
        logger.info("Running %d-iteration Monte Carlo bootstrap...", args.monte_carlo)
        mc_result = bootstrap_returns(
            backtest_df["strategy_returns"],
            n_simulations=args.monte_carlo,
            block_size=args.mc_block_size,
        )
        print_monte_carlo_report(mc_result, title="Bootstrap Monte Carlo")

    # --- save outputs ---
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    strat_tag = args.strategy.replace("-", "_")
    trade_log_file = output_dir / f"trade_log_{args.ticker.lower()}_{strat_tag}.csv"
    trade_log.to_csv(trade_log_file, index=False)
    logger.info("Saved: %s", trade_log_file)

    # equity curve vs buy-and-hold
    backtest_df["buy_hold_returns"] = backtest_df["close"].pct_change().fillna(0)
    backtest_df["buy_hold_equity"] = (1 + backtest_df["buy_hold_returns"]).cumprod()

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(
        backtest_df.index,
        backtest_df["equity_curve"],
        label=f"{strategy_label} Strategy",
        linewidth=1.2,
    )
    ax.plot(
        backtest_df.index,
        backtest_df["buy_hold_equity"],
        label=f"Buy & Hold {args.ticker}",
        linewidth=1.2,
    )
    ax.set_title(f"{args.ticker} {strategy_label} vs Buy-and-Hold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Equity")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()

    equity_plot_file = output_dir / f"equity_vs_buyhold_{args.ticker.lower()}_{strat_tag}.png"
    fig.savefig(equity_plot_file, dpi=150)
    plt.close(fig)
    logger.info("Saved: %s", equity_plot_file)

    # --- optional tearsheet ---
    if args.tearsheet:
        ts_path = output_dir / f"tearsheet_{args.ticker.lower()}_{strat_tag}.png"
        ts_fig = generate_tearsheet(
            backtest_df["strategy_returns"],
            benchmark=backtest_df["buy_hold_returns"],
            trade_log=trade_log,
            output_path=ts_path,
            title=f"{args.ticker} {strategy_label} tearsheet",
        )
        plt.close(ts_fig)
        logger.info("Saved tearsheet: %s", ts_path)

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
