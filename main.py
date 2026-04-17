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
from pathlib import Path
from functools import partial

import matplotlib.pyplot as plt

from src.data.loader import load_yahoo_ohlcv
from src.strategy.momentum import momentum_strategy
from src.strategy.mean_reversion import mean_reversion_strategy
from src.backtest.engine import backtest_strategy
from src.reporting.metrics import calculate_metrics
from src.risk.manager import RiskConfig, summarise_risk_events
from src.regime.detector import adaptive_strategy, RegimeConfig
from src.validation.walk_forward import (
    WalkForwardConfig,
    run_walk_forward,
    print_walk_forward_report,
)

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Systematic strategy backtester with risk management.",
    )

    # data
    parser.add_argument("--ticker", default="SPY", help="Yahoo Finance symbol (default: SPY)")
    parser.add_argument("--start", default="2010-01-01", help="Start date YYYY-MM-DD (default: 2010-01-01)")

    # strategy selection
    parser.add_argument(
        "--strategy", choices=["momentum", "mean-reversion", "adaptive"],
        default="momentum",
        help="Strategy type: momentum, mean-reversion, or adaptive (default: momentum)",
    )

    # momentum params
    parser.add_argument("--lookback", type=int, default=200, help="Momentum lookback period (default: 200)")
    parser.add_argument("--threshold", type=float, default=0.005, help="Signal threshold (default: 0.005)")
    parser.add_argument("--no-sma-filter", action="store_true", help="Disable SMA-200 regime filter")

    # mean reversion params
    parser.add_argument("--bb-window", type=int, default=20, help="Bollinger Bands window (default: 20)")
    parser.add_argument("--bb-std", type=float, default=2.0, help="Bollinger Bands std dev (default: 2.0)")
    parser.add_argument("--rsi-period", type=int, default=14, help="RSI period (default: 14)")
    parser.add_argument("--rsi-oversold", type=float, default=30.0, help="RSI oversold level (default: 30)")
    parser.add_argument("--rsi-overbought", type=float, default=70.0, help="RSI overbought level (default: 70)")
    parser.add_argument("--no-rsi-filter", action="store_true", help="Disable RSI filter for mean reversion")

    # regime detection (adaptive mode)
    parser.add_argument("--adx-period", type=int, default=14, help="ADX period for regime detection (default: 14)")
    parser.add_argument("--adx-threshold", type=float, default=25.0, help="ADX trending threshold (default: 25)")
    parser.add_argument("--hurst-window", type=int, default=100, help="Hurst exponent window (default: 100)")

    # vol targeting
    parser.add_argument("--vol-target", type=float, default=0.15, help="Annualised vol target (default: 0.15)")
    parser.add_argument("--vol-window", type=int, default=20, help="Realised vol window (default: 20)")

    # risk management
    parser.add_argument("--no-risk", action="store_true", help="Disable all risk controls")
    parser.add_argument("--stop-loss", type=float, default=0.05, help="Stop-loss threshold (default: 0.05)")
    parser.add_argument("--take-profit", type=float, default=0.10, help="Take-profit threshold (default: 0.10)")
    parser.add_argument("--trailing-stop", type=float, default=0.03, help="Trailing stop threshold (default: 0.03)")
    parser.add_argument("--max-position", type=float, default=1.0, help="Max position size (default: 1.0)")
    parser.add_argument("--daily-loss-limit", type=float, default=0.02, help="Daily loss limit (default: 0.02)")

    # walk-forward validation
    parser.add_argument("--walk-forward", action="store_true", help="Run walk-forward validation")
    parser.add_argument("--wf-is-days", type=int, default=504, help="Walk-forward in-sample days (default: 504)")
    parser.add_argument("--wf-oos-days", type=int, default=126, help="Walk-forward out-of-sample days (default: 126)")

    # costs
    parser.add_argument("--cost", type=float, default=0.001, help="Transaction cost (default: 0.001)")

    # output
    parser.add_argument("--output-dir", default="results", help="Output directory (default: results)")
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


def _build_strategy_fn(args: argparse.Namespace):
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
    elif args.strategy == "mean-reversion":
        return mr_fn
    else:
        return mom_fn


def _build_backtest_fn(args: argparse.Namespace, risk_config: RiskConfig | None):
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

    # --- trade log ---
    print("\n=== Trade Log (last 5) ===")
    if len(trade_log) > 0:
        print(trade_log.tail(5).to_string(index=False))
    else:
        print("  No trades found.")

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
    ax.plot(backtest_df.index, backtest_df["equity_curve"], label=f"{strategy_label} Strategy", linewidth=1.2)
    ax.plot(backtest_df.index, backtest_df["buy_hold_equity"], label=f"Buy & Hold {args.ticker}", linewidth=1.2)
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

    # summary table
    print("\n=== Backtest Data (last 10 rows) ===")
    cols_to_show = [
        "close", "signal", "position", "scaled_position",
        "strategy_returns", "equity_curve", "buy_hold_equity",
    ]
    existing_cols = [c for c in cols_to_show if c in backtest_df.columns]
    print(backtest_df[existing_cols].tail(10))


if __name__ == "__main__":
    main()
