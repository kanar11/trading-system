"""End-to-end demo exercising every major module.

The demo is *self-contained*: it generates synthetic OHLCV data so it
runs without network access and is suitable for CI smoke-tests. To
swap in real data, replace the synthetic-data block with a call to
``src.data.loader.load_yahoo_ohlcv`` or ``src.data.csv_loader.load_csv_ohlcv``.

It walks through:
    1. Data: synthesise multi-asset OHLCV.
    2. Strategy: run momentum, mean-reversion, breakout, and
       EMA-crossover signal generators on a single ticker.
    3. Ensemble: combine the four into one signal via majority vote.
    4. Backtest: run with risk controls + realistic execution costs.
    5. Validation: walk-forward, Monte Carlo bootstrap, Sharpe stats.
    6. Portfolio: aggregate per-ticker returns with min-variance and
       risk-parity optimisers.
    7. Reporting: tearsheet PNG + factor attribution vs the basket.

Run with:
    python examples/full_demo.py
"""

from __future__ import annotations

import logging
import sys
from functools import partial
from pathlib import Path

# make `python examples/full_demo.py` work without installing the package
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib

matplotlib.use("Agg")  # headless

import numpy as np
import pandas as pd

from src.backtest.engine import backtest_strategy
from src.execution.slippage import ExecutionConfig, apply_execution_costs
from src.portfolio import (
    PortfolioConfig,
    min_variance_weights,
    risk_parity_weights,
    run_portfolio_backtest,
)
from src.reporting.attribution import factor_regression, print_attribution_report
from src.reporting.metrics import calculate_metrics
from src.reporting.tearsheet import generate_tearsheet
from src.risk.manager import RiskConfig
from src.risk.sizing import kelly_fraction
from src.strategy.breakout import breakout_strategy
from src.strategy.ema_crossover import ema_crossover_strategy
from src.strategy.ensemble import majority_vote
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


def make_synthetic(name: str, seed: int, n: int = 1200) -> pd.DataFrame:
    """Synthesise a realistic-looking OHLCV frame."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2018-01-01", periods=n, freq="B")
    returns = rng.normal(0.0004, 0.012, n)
    close = 100 * np.exp(np.cumsum(returns))
    df = pd.DataFrame(
        {
            "open": close,
            "high": close * (1 + np.abs(rng.normal(0, 0.005, n))),
            "low": close * (1 - np.abs(rng.normal(0, 0.005, n))),
            "close": close,
            "volume": rng.integers(1_000_000, 10_000_000, n),
        },
        index=dates,
    )
    df.attrs["name"] = name
    return df


def section(title: str) -> None:
    print("\n" + "=" * 72)
    print(f"  {title}")
    print("=" * 72)


def main() -> None:
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        level=logging.INFO,
    )

    out_dir = Path("results/demo")
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- 1. data ---------------------------------------------------------
    section("1. Data -- synthetic 3-ticker basket")
    basket = {
        "AAA": make_synthetic("AAA", seed=1),
        "BBB": make_synthetic("BBB", seed=2),
        "CCC": make_synthetic("CCC", seed=3),
    }
    for t, df in basket.items():
        print(f"  {t}: {len(df)} rows  ({df.index[0].date()} -> {df.index[-1].date()})")
    main_df = basket["AAA"]

    # --- 2. strategies ---------------------------------------------------
    section("2. Strategies -- momentum / mean-rev / breakout / EMA cross")
    mom = momentum_strategy(main_df, lookback=60, threshold=0.0, use_sma_filter=False)["signal"]
    mr = mean_reversion_strategy(main_df, bb_window=20, bb_std=2.0)["signal"]
    bo = breakout_strategy(main_df, entry_window=20, exit_window=10)["signal"]
    em = ema_crossover_strategy(main_df, fast=20, slow=50)["signal"]

    counts = (
        pd.DataFrame({"momentum": mom, "mean_rev": mr, "breakout": bo, "ema_cross": em})
        .apply(lambda s: s.value_counts())
        .fillna(0)
        .astype(int)
    )
    print(counts.to_string())

    # --- 3. ensemble -----------------------------------------------------
    section("3. Ensemble -- majority vote across 4 strategies")
    signals_df = pd.DataFrame({"mom": mom, "mr": mr, "bo": bo, "em": em}).fillna(0).astype(int)
    ensemble_sig = majority_vote(signals_df)
    ensemble_df = main_df.copy()
    ensemble_df["signal"] = ensemble_sig
    print(f"  Ensemble signal distribution: {dict(ensemble_sig.value_counts())}")

    # --- 4. backtest with risk + execution costs -------------------------
    section("4. Backtest -- risk controls + realistic execution")
    risk = RiskConfig(stop_loss=0.05, take_profit=0.10, trailing_stop=0.03)
    bt, trades = backtest_strategy(
        ensemble_df,
        transaction_cost=0.0,
        vol_target=None,
        risk_config=risk,
    )
    # impact_coeff and participation_cap tuned for this demo so a full
    # long->short reversal (|delta|=2) does not eat the equity curve
    bt = apply_execution_costs(
        bt,
        ExecutionConfig(
            spread_bps=5,
            impact_coeff=0.01,
            participation_cap=4.0,
            fixed_cost_per_trade=0.00002,
        ),
    )
    base_metrics = calculate_metrics(bt["strategy_returns"])
    for k, v in base_metrics.items():
        print(f"  {k:<18}{v:>+8.4f}")
    print(f"  Closed trades:    {len(trades)}")

    # --- 5. validation ---------------------------------------------------
    # Walk-forward expects strategy_fn to *generate* signals on each window;
    # use a plain momentum strategy rather than the pre-computed ensemble.
    section("5a. Walk-forward (momentum strategy)")
    wf = run_walk_forward(
        main_df,
        strategy_fn=partial(momentum_strategy, lookback=60, threshold=0.0, use_sma_filter=False),
        backtest_fn=partial(backtest_strategy, transaction_cost=0.001, vol_target=None),
        config=WalkForwardConfig(in_sample_days=504, out_of_sample_days=126),
    )
    print_walk_forward_report(wf)

    section("5b. Monte Carlo bootstrap (i.i.d.)")
    mc = bootstrap_returns(bt["strategy_returns"], n_simulations=500, block_size=1)
    print_monte_carlo_report(mc, title="iid bootstrap")

    section("5c. Sharpe significance + Deflated SR")
    sr_test = sharpe_ttest(bt["strategy_returns"])
    psr = probabilistic_sharpe_ratio(bt["strategy_returns"])
    dsr = deflated_sharpe_ratio(bt["strategy_returns"], n_trials=50)
    print(
        f"  Annualised SR     : {sr_test.sharpe_annualised:>+7.2f}  "
        f"(t={sr_test.t_stat:+.2f}, p={sr_test.p_value_two_sided:.3f})"
    )
    print(f"  PSR(SR > 0)       : {psr:.3f}")
    print(f"  Deflated SR (50 trials): {dsr:.3f}")

    # --- 6. portfolio ----------------------------------------------------
    section("6. Portfolio -- equal vs min-variance vs risk-parity")
    strategy_fn = partial(momentum_strategy, lookback=60, threshold=0.0, use_sma_filter=False)
    backtest_fn = partial(backtest_strategy, transaction_cost=0.001, vol_target=None)

    equal_res = run_portfolio_backtest(
        basket, strategy_fn, backtest_fn, PortfolioConfig(weighting="equal")
    )
    invvol_res = run_portfolio_backtest(
        basket, strategy_fn, backtest_fn, PortfolioConfig(weighting="inverse_vol")
    )

    # plug closed-form optimisers using the per-asset return panel
    minvar_w = min_variance_weights(equal_res.returns)
    rp_w = risk_parity_weights(equal_res.returns)
    print(f"  Min-variance weights:  {minvar_w.round(3).to_dict()}")
    print(f"  Risk-parity weights:   {rp_w.round(3).to_dict()}")
    print(f"  Equal portfolio Sharpe:        {equal_res.metrics['Sharpe Ratio']:+.2f}")
    print(f"  Inverse-vol portfolio Sharpe:  {invvol_res.metrics['Sharpe Ratio']:+.2f}")

    # apply optimiser weights manually for comparison
    for name, w in (("min-variance", minvar_w), ("risk-parity", rp_w)):
        port_ret = (equal_res.returns * w).sum(axis=1)
        m = calculate_metrics(port_ret)
        print(f"  {name:<18} portfolio Sharpe:  {m['Sharpe Ratio']:+.2f}")

    # --- 7. reporting ----------------------------------------------------
    section("7a. Tearsheet PNG")
    tearsheet_path = out_dir / "tearsheet.png"
    generate_tearsheet(
        bt["strategy_returns"],
        benchmark=bt["market_returns"],
        trade_log=trades,
        output_path=tearsheet_path,
        title="Full-demo ensemble strategy",
    )
    print(f"  Saved {tearsheet_path}  ({tearsheet_path.stat().st_size} bytes)")

    section("7b. Factor attribution vs the basket")
    # use the per-asset returns as "factors" for attribution
    attribution = factor_regression(
        bt["strategy_returns"],
        factors=equal_res.returns,
    )
    print_attribution_report(attribution, title="Ensemble vs basket factors")

    # --- 8. position sizing example -------------------------------------
    section("8. Kelly sizing on trade returns")
    if not trades.empty:
        kelly_f = kelly_fraction(trades["trade_return"], cap=1.0)
        print(f"  Half-Kelly position-size estimate: {kelly_f:.3f} of equity")

    section("Demo complete")
    print(f"  Outputs written to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
