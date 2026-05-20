# Systematic Trading Research Backtester

A Python research project for designing, testing, and evaluating systematic trading strategies on market data.

This project demonstrates a full research workflow: data ingestion, signal generation, cost-aware backtesting with risk management, performance evaluation, trade-level analytics, parameter optimisation, and walk-forward validation.

## What the project does

The system currently:

- downloads historical OHLCV data using `yfinance` (any ticker)
- generates trading signals using four strategies: momentum, mean reversion, Donchian breakout, and adaptive (regime-based)
- detects market regimes (trending vs mean-reverting) using ADX and Hurst exponent
- applies a cost-aware backtest with transaction costs and volatility targeting
- models realistic execution costs (bid-ask spread + square-root market impact + fixed commission)
- enforces risk management rules (stop-loss, take-profit, trailing stop, position limits, daily loss limit)
- offers position-sizing helpers: fractional Kelly, ATR-based and fixed-fractional sizing
- runs walk-forward validation to test strategy robustness (rolling IS/OOS)
- runs Monte Carlo robustness analysis (bootstrap and trade-shuffle) for confidence intervals
- aggregates single-asset strategies into a multi-asset portfolio (equal-weight, inverse-vol, custom, min-variance, max-Sharpe, or risk-parity weights)
- runs factor / attribution regression to separate alpha from passive factor exposure
- generates multi-panel tear-sheet reports (equity, drawdown, rolling Sharpe, monthly heatmap, distribution)
- computes trade-level analytics (win rate, profit factor, expectancy, streaks, payoff ratio)
- builds equity curves and saves them as plots
- calculates performance statistics (Sharpe, Sortino, Calmar, CAGR, Max Drawdown)
- exports trade logs and parameter sweep results

## Project structure

```text
trading_system/
├── src/
│   ├── data/
│   │   └── loader.py              # Yahoo Finance data download
│   ├── strategy/
│   │   ├── momentum.py            # Momentum signal generator
│   │   ├── mean_reversion.py      # Bollinger Bands + RSI strategy
│   │   └── breakout.py            # Donchian channel breakout (turtle-style)
│   ├── backtest/
│   │   └── engine.py              # Cost-aware backtest engine
│   ├── execution/
│   │   └── slippage.py            # Spread + sqrt-impact execution model
│   ├── risk/
│   │   ├── manager.py             # Risk management controls
│   │   └── sizing.py              # Kelly / ATR / fixed-fractional sizing
│   ├── regime/
│   │   └── detector.py            # Market regime detection (ADX + Hurst)
│   ├── validation/
│   │   ├── walk_forward.py        # Walk-forward validation framework
│   │   └── monte_carlo.py         # Bootstrap + trade-shuffle robustness
│   ├── portfolio/
│   │   ├── portfolio.py           # Multi-asset portfolio backtest
│   │   └── optimizer.py           # Min-variance / max-Sharpe / risk-parity weights
│   └── reporting/
│       ├── metrics.py             # Portfolio metrics + trade analytics
│       ├── plots.py               # Equity curve plotting
│       ├── trades.py              # Trade log builder (standalone utility)
│       ├── sweep.py               # Parameter sweep runner
│       ├── tearsheet.py           # Multi-panel PNG tear-sheet report
│       └── attribution.py         # Factor / alpha regression
├── tests/
│   ├── conftest.py                # Shared fixtures (OHLCV, returns)
│   ├── test_metrics.py            # Portfolio + trade-level metric tests
│   ├── test_risk_manager.py       # Risk controls tests
│   ├── test_momentum.py           # Momentum strategy tests
│   ├── test_mean_reversion.py     # Mean reversion strategy tests
│   ├── test_walk_forward.py       # Walk-forward validation tests
│   ├── test_monte_carlo.py        # Monte Carlo robustness tests
│   ├── test_regime.py             # Regime detection tests
│   ├── test_breakout.py           # Donchian breakout tests
│   ├── test_sizing.py             # Position-sizing tests
│   ├── test_portfolio.py          # Portfolio backtest tests
│   ├── test_optimizer.py          # Portfolio optimiser tests
│   ├── test_execution.py          # Execution-cost model tests
│   ├── test_tearsheet.py          # Tear-sheet generator tests
│   ├── test_attribution.py        # Factor attribution tests
│   └── test_engine.py             # Backtest engine tests
├── main.py                        # Main pipeline entry point (CLI)
├── grid_search.py                 # Grid search script
├── plot_heatmap.py                # Heatmap visualisation
├── pyproject.toml                 # Project config and dependencies
├── requirements.txt
└── README.md
```

## How to run

### 1. Create and activate a virtual environment

Mac / Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the project

```bash
# default: momentum strategy on SPY
python main.py

# mean reversion strategy on AAPL
python main.py --strategy mean-reversion --ticker AAPL

# adaptive (regime-based) strategy
python main.py --strategy adaptive

# custom momentum parameters
python main.py --lookback 100 --threshold 0.01 --no-sma-filter

# walk-forward validation
python main.py --walk-forward

# walk-forward with mean reversion and custom windows
python main.py --strategy mean-reversion --walk-forward --wf-is-days 252 --wf-oos-days 63

# disable risk management
python main.py --no-risk

# verbose logging
python main.py -v
```

### 4. Run tests

```bash
pytest tests/ -v
```

## Strategies

### Momentum

Generates signals based on lookback returns. Goes long when the lookback return exceeds the threshold, short when it falls below. An optional SMA-200 regime filter restricts long signals to periods when price is above its 200-day moving average (and vice versa for shorts).

```bash
python main.py --strategy momentum --lookback 200 --threshold 0.005
```

Key parameters: `--lookback` (period for return calculation), `--threshold` (minimum return to trigger signal), `--no-sma-filter` (disable SMA-200 filter).

### Mean Reversion

Uses Bollinger Bands to detect overbought/oversold conditions. Enters when price moves outside the bands and exits when it reverts to the middle band. RSI acts as a confirmation filter to reduce false entries.

```bash
python main.py --strategy mean-reversion --bb-window 20 --bb-std 2.0 --rsi-period 14
```

Key parameters: `--bb-window` (Bollinger Bands window), `--bb-std` (standard deviations), `--rsi-period`, `--rsi-oversold`, `--rsi-overbought`, `--no-rsi-filter`.

### Donchian Breakout

Classic turtle-style trend follower. Enters long when price prints a new N-day high and short on a new N-day low; exits when price crosses an opposite shorter-window channel. An optional ATR filter suppresses signals when the breakout size is small relative to recent volatility.

```python
from src.strategy.breakout import breakout_strategy

signals = breakout_strategy(
    df,
    entry_window=20,    # N-day high/low for entry
    exit_window=10,     # M-day opposite channel for exit
    atr_period=14,
    atr_filter=0.5,     # require breakout >= 0.5 * ATR
    allow_short=True,
)
```

### Adaptive (Regime-Based)

Automatically detects the market regime and selects the appropriate strategy. Uses ADX (trend strength) and Hurst exponent (mean-reversion tendency) to classify each day as trending, mean-reverting, or undefined. Applies momentum during trends, mean reversion during ranges, and goes flat when the regime is unclear. A smoothing window prevents rapid switching.

```bash
python main.py --strategy adaptive --adx-threshold 25 --hurst-window 100
```

Key parameters: `--adx-period`, `--adx-threshold`, `--hurst-window`.

## Regime detection

The regime detector (`src/regime/detector.py`) classifies each day into one of three regimes using two independent indicators:

- **ADX** (Average Directional Index) measures trend strength regardless of direction. Values above 25 indicate a strong trend.
- **Hurst exponent** estimates persistence in the price series. H > 0.55 suggests trending behaviour; H < 0.45 suggests mean reversion; H near 0.5 is a random walk.

The final classification requires both indicators to agree. A rolling majority-vote smoothing window prevents whipsawing between regimes. Volatility regime (high/low) is also computed as a secondary signal.

## Risk management

Risk controls are applied as a middleware layer between position sizing and return calculation. Each rule can be independently enabled or disabled.

```python
from src.risk.manager import RiskConfig

risk_config = RiskConfig(
    stop_loss=0.05,        # exit if loss exceeds 5%
    take_profit=0.10,      # exit if profit reaches 10%
    trailing_stop=0.03,    # exit if 3% drawdown from peak
    max_position=1.0,      # cap position size at 100%
    daily_loss_limit=0.02, # flatten all if daily loss exceeds 2%
)
```

CLI flags: `--stop-loss`, `--take-profit`, `--trailing-stop`, `--max-position`, `--daily-loss-limit`, `--no-risk`.

## Position sizing

`src/risk/sizing.py` provides three independent sizing helpers that can be used at calibration time (fit on IS, apply on OOS) or as post-hoc studies:

- **`kelly_fraction(returns, cap, kelly_fraction_of_full)`** — continuous Kelly `f* = mean / variance`. Half-Kelly is the default to control variance.
- **`atr_position_size(price, atr, equity, risk_per_trade, atr_multiple, max_size)`** — sets notional so that hitting an `N * ATR` stop costs exactly `risk_per_trade` of equity.
- **`fixed_fractional(win_rate, payoff_ratio, cap)`** — discrete-outcome Kelly from trade-log stats: `f* = W - (1 - W) / R`.

All helpers return a position fraction in `[0, cap]` and degrade gracefully (return `0.0`) when the edge is non-positive.

## Execution / slippage model

`src/execution/slippage.py` replaces the engine's flat `transaction_cost` with a reduced-form model that captures the qualitative shape of real-world execution costs:

```
cost_fraction = 0.5 * spread_bps / 10_000
              + impact_coeff * (|delta| / participation_cap) ** impact_exponent
              + fixed_cost_per_trade
```

`spread_bps` covers the round-trip bid-ask spread, the impact term implements the square-root law (`impact_exponent=0.5` by default), and `fixed_cost_per_trade` covers per-trade commission. Apply it to an engine output via:

```python
from src.execution import ExecutionConfig, apply_execution_costs

bt = apply_execution_costs(bt, ExecutionConfig(spread_bps=5, impact_coeff=0.1))
```

This recomputes `transaction_cost`, `strategy_returns` and `equity_curve` in place.

## Walk-forward validation

Tests strategy robustness by splitting data into rolling in-sample (training) and out-of-sample (testing) windows. For each fold, the strategy runs on the full IS+OOS window to avoid warm-up artefacts, but only the OOS portion is evaluated. Reports per-fold metrics, combined OOS equity, and IS-vs-OOS Sharpe degradation.

```bash
python main.py --walk-forward --wf-is-days 504 --wf-oos-days 126
```

A Sharpe degradation above 50% suggests the strategy may be overfitting to in-sample data. Below 20% indicates good robustness.

## Monte Carlo robustness

Walk-forward measures degradation across time; Monte Carlo measures degradation under reshuffling. Two routines in `src/validation/monte_carlo.py`:

- **`bootstrap_returns(returns, n_simulations, block_size)`** — resamples the daily return series with replacement and recomputes Sharpe, max drawdown, total return, etc. Use `block_size > 1` for a moving-block bootstrap that preserves short-range autocorrelation.
- **`shuffle_trade_log(trade_returns, n_simulations)`** — permutes the order of completed trades without replacement. The set of trades is identical to the original, so this isolates path-dependence: a wide spread in max drawdown across permutations suggests the in-sample drawdown is partly an ordering artefact.

Both return a `MonteCarloResult` with per-simulation metrics and a mean / std / 5%/50%/95% summary table. Use `print_monte_carlo_report(result)` for a formatted dump.

## Multi-asset portfolio

`src/portfolio/portfolio.py` runs the same single-asset strategy across a basket of tickers and aggregates the per-asset return streams into a portfolio using one of three weighting schemes:

- **`equal`** — flat 1/N weights, rebalanced daily.
- **`inverse_vol`** — weights inversely proportional to trailing realised volatility (simple risk-parity proxy).
- **`custom`** — user-supplied static weights, normalised to sum to 1.

For covariance-aware allocation, `src/portfolio/optimizer.py` provides three closed-form / iterative schemes (no scipy required):

- **`min_variance_weights(returns, cov=None)`** — long-only minimum-variance portfolio via `Σ⁻¹ 1`.
- **`max_sharpe_weights(returns, cov=None, rf_daily=0)`** — long-only tangency portfolio via `Σ⁻¹ (μ − rf)`.
- **`risk_parity_weights(returns, cov=None)`** — true equal risk-contribution weights via cyclical coordinate descent (Maillard, Roncalli & Teïletche 2010).

All optimisers clip negative weights to zero and re-normalise.

```python
from src.portfolio import PortfolioConfig, run_portfolio_backtest

basket = {"SPY": spy_df, "QQQ": qqq_df, "GLD": gld_df}
result = run_portfolio_backtest(
    basket, strategy_fn, backtest_fn,
    config=PortfolioConfig(weighting="inverse_vol", vol_window=20),
)
print(result.metrics)        # portfolio-level Sharpe / MaxDD / etc.
print(result.per_asset_metrics)  # one dict per ticker
```

## Factor attribution

`src/reporting/attribution.py` regresses daily strategy returns against one or more factor return streams:

```
r_strategy_t = alpha + Σ_i beta_i * f_{i,t} + epsilon_t
```

The intercept is the annualised excess return that *cannot* be explained by the supplied factors. Per-factor t-stats, R² and residuals are reported. Useful for asking "is my edge really alpha, or just a hidden momentum / market beta?"

```python
from src.reporting.attribution import factor_regression, print_attribution_report

result = factor_regression(strategy_returns, factors=ff3_df, rf_rate=0.0)
print_attribution_report(result)
```

Pure OLS via `numpy.linalg.lstsq` — no scipy needed.

## Tear-sheet report

`src/reporting/tearsheet.py` generates a single-PNG, multi-panel report with the equity curve (vs an optional benchmark), underwater drawdown, rolling Sharpe, monthly-returns heatmap, return distribution, and a performance summary table.

```python
from src.reporting.tearsheet import generate_tearsheet

fig = generate_tearsheet(
    strategy_returns,
    benchmark=spy_returns,
    trade_log=trade_log,
    output_path="results/tearsheet.png",
    title="SPY momentum tear-sheet",
)
```

Returns a matplotlib `Figure` and writes a PNG. Headless-friendly — safe to call from CI.

## Trade-level analytics

The system tracks every round-trip trade and computes detailed statistics:

- **Win rate** — fraction of profitable trades
- **Profit factor** — gross wins divided by gross losses
- **Expectancy** — average return per trade
- **Payoff ratio** — average win divided by average loss
- **Largest win / loss** — best and worst single trades
- **Win/loss streaks** — longest consecutive winning and losing runs
- **Holding period** — average days per trade (overall, winners, losers)
- **Direction breakdown** — long vs short trade counts

Trade stats are printed automatically after each backtest run and are included in parameter sweep CSV output (win rate, profit factor, expectancy).

## Performance metrics

Portfolio-level metrics computed from the daily return series:

- **Total Return** — cumulative return over the period
- **CAGR** — compound annual growth rate
- **Sharpe Ratio** — risk-adjusted return (annualised)
- **Sortino Ratio** — downside-risk-adjusted return
- **Max Drawdown** — largest peak-to-trough decline
- **Calmar Ratio** — CAGR divided by max drawdown

## Parameter sweep

The parameter sweep runner tests a grid of lookback/threshold combinations and exports ranked results to CSV. Results now include trade-level stats alongside portfolio metrics.

```bash
python grid_search.py
```

Results are saved to `results/sweep_results.csv` and the top 10 configurations are printed to the console.

## Limitations

This is a research prototype, not a production trading system. Current limitations include Yahoo Finance data only, simplified regime detection (not a hidden Markov model), execution-cost model that is reduced-form (no order-book simulation, no queue position), portfolio optimisers that clip negative weights rather than solving a constrained QP, and Monte Carlo resampling that assumes returns are stationary.

## Tech stack

- Python 3.11+
- pandas, NumPy
- matplotlib, seaborn
- yfinance
- pytest

## Author

Built as a personal quant / systematic trading research project.
