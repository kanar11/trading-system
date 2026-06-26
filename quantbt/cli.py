"""Command-line interface for the backtesting pipeline.

A run is described by a :class:`~quantbt.config.RunConfig`. The CLI builds one
from an optional ``--config`` YAML file plus flag overrides (flags win), then
dispatches the chosen strategy through the registry. The fully resolved config
is written next to the results for reproducibility.

Usage:
    quantbt-backtest
    quantbt-backtest --strategy mean-reversion --ticker AAPL
    quantbt-backtest --config configs/example.yaml
    quantbt-backtest --strategy adaptive --walk-forward
"""

from __future__ import annotations

import argparse
import logging
from functools import partial
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import yaml

from quantbt.backtest.engine import backtest_strategy
from quantbt.config import RunConfig
from quantbt.data.loader import load_yahoo_ohlcv
from quantbt.reporting.metrics import calculate_metrics, calculate_trade_stats
from quantbt.risk.manager import summarise_risk_events
from quantbt.strategy import available, build_strategy
from quantbt.validation.walk_forward import print_walk_forward_report, run_walk_forward

logger = logging.getLogger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments. All override flags default to None (unset)."""
    parser = argparse.ArgumentParser(
        description="Systematic strategy backtester with risk management.",
    )
    parser.add_argument("--config", help="Path to a YAML config file (base values)")

    # data
    parser.add_argument("--ticker", help="Yahoo Finance symbol (default: SPY)")
    parser.add_argument("--start", help="Start date YYYY-MM-DD (default: 2010-01-01)")
    parser.add_argument("--end", help="End date YYYY-MM-DD (default: today)")

    # strategy selection
    parser.add_argument("--strategy", choices=available(), help="Strategy type")

    # momentum params
    parser.add_argument("--lookback", type=int, help="Momentum lookback period")
    parser.add_argument("--threshold", type=float, help="Momentum signal threshold")
    parser.add_argument(
        "--no-sma-filter", action="store_true", default=None, help="Disable SMA-200 filter"
    )

    # mean reversion params
    parser.add_argument("--bb-window", type=int, help="Bollinger Bands window")
    parser.add_argument("--bb-std", type=float, help="Bollinger Bands std dev")
    parser.add_argument("--rsi-period", type=int, help="RSI period")
    parser.add_argument("--rsi-oversold", type=float, help="RSI oversold level")
    parser.add_argument("--rsi-overbought", type=float, help="RSI overbought level")
    parser.add_argument(
        "--no-rsi-filter", action="store_true", default=None, help="Disable RSI filter"
    )

    # regime detection (adaptive mode)
    parser.add_argument("--adx-period", type=int, help="ADX period")
    parser.add_argument("--adx-threshold", type=float, help="ADX trending threshold")
    parser.add_argument("--hurst-window", type=int, help="Hurst exponent window")

    # vol targeting
    parser.add_argument("--vol-target", type=float, help="Annualised vol target")
    parser.add_argument("--vol-window", type=int, help="Realised vol window")

    # risk management
    parser.add_argument(
        "--no-risk", action="store_true", default=None, help="Disable all risk controls"
    )
    parser.add_argument("--stop-loss", type=float, help="Stop-loss threshold")
    parser.add_argument("--take-profit", type=float, help="Take-profit threshold")
    parser.add_argument("--trailing-stop", type=float, help="Trailing stop threshold")
    parser.add_argument("--max-position", type=float, help="Max position size")
    parser.add_argument("--daily-loss-limit", type=float, help="Daily loss limit")

    # walk-forward validation
    parser.add_argument(
        "--walk-forward", action="store_true", default=None, help="Run walk-forward validation"
    )
    parser.add_argument("--wf-is-days", type=int, help="Walk-forward in-sample days")
    parser.add_argument("--wf-oos-days", type=int, help="Walk-forward out-of-sample days")

    # costs
    parser.add_argument("--cost", type=float, help="Transaction cost")

    # output
    parser.add_argument("--output-dir", help="Output directory (default: results)")
    parser.add_argument("-v", "--verbose", action="store_true", default=None, help="Debug logging")

    return parser.parse_args(argv)


def _set(target: dict[str, Any], section: str, key: str, value: Any) -> None:
    """Set ``target[section][key] = value`` when value is not None."""
    if value is not None:
        target.setdefault(section, {})[key] = value


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge ``override`` into ``base`` (override wins)."""
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def build_config(args: argparse.Namespace) -> RunConfig:
    """Build a validated RunConfig from a YAML base plus CLI overrides."""
    overrides: dict[str, Any] = {}
    if args.strategy is not None:
        overrides["strategy"] = args.strategy

    _set(overrides, "data", "ticker", args.ticker)
    _set(overrides, "data", "start", args.start)
    _set(overrides, "data", "end", args.end)

    _set(overrides, "momentum", "lookback", args.lookback)
    _set(overrides, "momentum", "threshold", args.threshold)
    if args.no_sma_filter:
        _set(overrides, "momentum", "use_sma_filter", False)

    _set(overrides, "mean_reversion", "bb_window", args.bb_window)
    _set(overrides, "mean_reversion", "bb_std", args.bb_std)
    _set(overrides, "mean_reversion", "rsi_period", args.rsi_period)
    _set(overrides, "mean_reversion", "rsi_oversold", args.rsi_oversold)
    _set(overrides, "mean_reversion", "rsi_overbought", args.rsi_overbought)
    if args.no_rsi_filter:
        _set(overrides, "mean_reversion", "use_rsi_filter", False)

    _set(overrides, "regime", "adx_period", args.adx_period)
    _set(overrides, "regime", "adx_trending_threshold", args.adx_threshold)
    _set(overrides, "regime", "hurst_window", args.hurst_window)

    _set(overrides, "backtest", "vol_target", args.vol_target)
    _set(overrides, "backtest", "vol_window", args.vol_window)
    _set(overrides, "backtest", "transaction_cost", args.cost)

    if args.no_risk:
        _set(overrides, "risk", "enabled", False)
    _set(overrides, "risk", "stop_loss", args.stop_loss)
    _set(overrides, "risk", "take_profit", args.take_profit)
    _set(overrides, "risk", "trailing_stop", args.trailing_stop)
    _set(overrides, "risk", "max_position", args.max_position)
    _set(overrides, "risk", "daily_loss_limit", args.daily_loss_limit)

    if args.walk_forward:
        _set(overrides, "walk_forward", "enabled", True)
    _set(overrides, "walk_forward", "in_sample_days", args.wf_is_days)
    _set(overrides, "walk_forward", "out_of_sample_days", args.wf_oos_days)

    _set(overrides, "output", "output_dir", args.output_dir)
    if args.verbose:
        _set(overrides, "output", "verbose", True)

    if args.config is not None:
        base = yaml.safe_load(Path(args.config).read_text(encoding="utf-8")) or {}
        if not isinstance(base, dict):
            raise ValueError(f"Config file {args.config} must contain a YAML mapping.")
        return RunConfig(**_deep_merge(base, overrides))
    return RunConfig(**overrides)


def setup_logging(verbose: bool = False) -> None:
    """Configure logging format and level."""
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        level=logging.DEBUG if verbose else logging.INFO,
    )


def _print_trade_stats(trade_stats: dict[str, Any]) -> None:
    """Print the trade-level analytics block."""
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
        print(f"  Long/Short:      {trade_stats['Long Trades']}L / {trade_stats['Short Trades']}S")


def run(config: RunConfig) -> None:
    """Execute a full run (backtest or walk-forward) from a config."""
    output_dir = Path(config.output.output_dir)
    output_dir.mkdir(exist_ok=True)
    config.to_yaml(output_dir / "run_config.yaml")

    label = config.strategy_label()
    ticker = config.data.ticker

    logger.info("Loading %s data from %s...", ticker, config.data.start)
    df = load_yahoo_ohlcv(ticker=ticker, start=config.data.start, end=config.data.end)
    logger.info("Loaded %d rows.", len(df))

    strategy_fn = build_strategy(config)
    backtest_fn = partial(
        backtest_strategy,
        transaction_cost=config.backtest.transaction_cost,
        vol_target=config.backtest.vol_target,
        vol_window=config.backtest.vol_window,
        risk_config=config.risk.to_dataclass(),
    )

    if config.walk_forward.enabled:
        logger.info("Running walk-forward validation (%s)...", label)
        results = run_walk_forward(df, strategy_fn, backtest_fn, config.walk_forward.to_dataclass())
        print_walk_forward_report(results)

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(results["oos_equity"].index, results["oos_equity"].values, linewidth=1.2)
        ax.set_title(f"{ticker} {label} — Walk-Forward OOS Equity")
        ax.set_xlabel("Date")
        ax.set_ylabel("Equity")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        wf_plot = output_dir / f"walk_forward_oos_{ticker.lower()}.png"
        fig.savefig(wf_plot, dpi=150)
        plt.close(fig)
        logger.info("Saved: %s", wf_plot)
        return

    logger.info("Generating %s signals...", label)
    strategy_df = strategy_fn(df)

    logger.info("Running backtest...")
    backtest_df, trade_log = backtest_fn(strategy_df)

    metrics = calculate_metrics(backtest_df["strategy_returns"])
    metrics["Trades (closed)"] = len(trade_log)

    print(f"\n=== {label} Strategy Metrics ===")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}")

    if config.risk.enabled:
        risk_events = summarise_risk_events(backtest_df)
        if risk_events:
            print("\n=== Risk Events ===")
            for event, count in risk_events.items():
                print(f"  {event}: {count}")

    trade_stats = calculate_trade_stats(trade_log)
    if trade_stats["Total Trades"] > 0:
        _print_trade_stats(trade_stats)

    print("\n=== Trade Log (last 5) ===")
    print(trade_log.tail(5).to_string(index=False) if len(trade_log) > 0 else "  No trades found.")

    strat_tag = config.strategy.replace("-", "_")
    trade_log_file = output_dir / f"trade_log_{ticker.lower()}_{strat_tag}.csv"
    trade_log.to_csv(trade_log_file, index=False)
    logger.info("Saved: %s", trade_log_file)

    backtest_df["buy_hold_returns"] = backtest_df["close"].pct_change().fillna(0)
    backtest_df["buy_hold_equity"] = (1 + backtest_df["buy_hold_returns"]).cumprod()

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(
        backtest_df.index, backtest_df["equity_curve"], label=f"{label} Strategy", linewidth=1.2
    )
    ax.plot(
        backtest_df.index,
        backtest_df["buy_hold_equity"],
        label=f"Buy & Hold {ticker}",
        linewidth=1.2,
    )
    ax.set_title(f"{ticker} {label} vs Buy-and-Hold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Equity")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    equity_plot_file = output_dir / f"equity_vs_buyhold_{ticker.lower()}_{strat_tag}.png"
    fig.savefig(equity_plot_file, dpi=150)
    plt.close(fig)
    logger.info("Saved: %s", equity_plot_file)


def main(argv: list[str] | None = None) -> None:
    """Entry point: parse args, build config, run."""
    args = parse_args(argv)
    config = build_config(args)
    setup_logging(config.output.verbose)
    run(config)


if __name__ == "__main__":
    main()
