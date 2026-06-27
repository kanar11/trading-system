# Systematic Trading Research Platform

A Python research platform for designing, testing, and evaluating systematic trading strategies on market data — modelled after professional quant-trading stacks.

This project is structured around two complementary backtesting modes:

1. **Vectorised mode** (`src.backtest.engine`) — signal-in / equity-out, blazing fast, ideal for parameter sweeps and walk-forward research.
2. **Event-driven mode** (`src.backtest.event_engine`) — bar-by-bar simulation with a full Order Management System (limit / stop / stop-limit orders, partial fills, commission + slippage, cost-basis position tracking). Same API surface as the paper-broker live adapter, so strategies port straight to a live broker without rewrites.

## Architecture

```
                       ┌──────────────────────────┐
                       │      Strategy            │
                       │  on_bar / signal funcs   │
                       └─────────────┬────────────┘
                                     │
                  ┌──────────────────┴──────────────────┐
                  │                                     │
        ┌─────────▼──────────┐              ┌───────────▼─────────┐
        │ Vectorised engine  │              │ Event-driven engine │
        │ + Risk + Execution │              │  + OMS + Slippage   │
        └─────────┬──────────┘              └───────────┬─────────┘
                  │                                     │
                  └──────────────────┬──────────────────┘
                                     │
                       ┌─────────────▼────────────┐
                       │     Portfolio / OMS      │
                       │ cash + positions + PnL   │
                       └─────────────┬────────────┘
                                     │
                       ┌─────────────▼────────────┐
                       │  Reporting + Validation  │
                       │ metrics / tearsheet /    │
                       │ MC bootstrap / WF / DSR  │
                       └──────────────────────────┘
```

Plug-in points: indicators (`src.indicators`), position sizing (`src.risk.sizing`), risk metrics (`src.risk.metrics`), portfolio optimisation (`src.portfolio.optimizer`), and a paper broker (`src.live.broker`) that shares the same OMS as the event engine.

## What the project does

The system currently:

- downloads OHLCV data via `yfinance` or local CSV, with a transparent parquet-backed cache (`src.data.cache`)
- ships pre-defined universes (FAANG, Dow 30, sector ETFs, benchmarks, factor ETFs) in `src.data.universe`
- aggregates intraday bars to any frequency via `src.data.resample` (1m → 5m → 1h → 1D → 1W → 1ME)
- audits OHLCV data quality — duplicate timestamps, unsorted index, missing values, OHLC inconsistencies, extreme returns, stale-price runs — and conservatively cleans it (`src.data.quality`)
- maintains a comprehensive technical-indicators library (`src.indicators`): SMA / EMA / WMA / VWMA, RSI, MACD, Stochastic, Williams %R, CCI, ROC, ATR (SMA/EMA/Wilder smoothings), Bollinger, Keltner, Donchian, OBV, anchored VWAP, Chaikin A/D, Hull MA, Aroon, TRIX, CMO, MFI
- generates trading signals using eight strategy templates: momentum, mean reversion, Donchian breakout, EMA crossover, MACD, TRIX, pairs (cointegration), and adaptive (regime-based)
- combines multiple strategies via majority vote, weighted sum or unanimous-consent ensemble combiners
- detects market regimes (trending vs mean-reverting) using ADX and Hurst exponent, or a data-driven Gaussian HMM (Baum-Welch EM + Viterbi, pure numpy)
- runs **vectorised** backtests with transaction costs, volatility targeting and a risk middleware (stop-loss, take-profit, trailing stop, position limits, daily loss limit)
- runs **event-driven** backtests through a full OMS — MARKET / LIMIT / STOP / STOP_LIMIT orders, DAY / GTC / IOC / FOK TIF, intrabar limit matching, gap-safe stop fills, partial fills, weighted-avg cost basis, realized vs unrealized PnL splits
- models realistic execution costs (bid-ask spread + square-root market impact + fixed commission), plus Almgren-Chriss optimal-execution scheduling and participation-rate impact
- offers position-sizing helpers: fractional Kelly, ATR-based, fixed-fractional, volatility-target, CPPI and drawdown-throttle sizing
- exposes a `Broker` interface with a `PaperBroker` implementation that shares the same OMS — a clean seam for future IB / Alpaca / Binance adapters
- computes **25+ performance metrics** including Sharpe, Sortino, Calmar, CAGR, max drawdown, **Value-at-Risk** (historical / parametric), **Conditional VaR**, **Omega ratio**, **Ulcer Index**, **gain-to-pain**, **drawdown duration & recovery time**, **tail ratio**, **downside/upside deviation**, **rolling beta vs benchmark**, **skew / kurtosis**, **tracking error & information ratio**, **Sterling / Burke ratios**
- runs walk-forward validation, Monte Carlo bootstrap, trade-shuffle robustness and statistical Sharpe significance tests (t-test, Probabilistic SR, **Deflated SR** for multiple-testing correction), plus a CSCV **Probability of Backtest Overfitting** estimate
- aggregates single-asset strategies into a multi-asset portfolio (equal-weight, inverse-vol, custom, min-variance, max-Sharpe, risk-parity, maximum-diversification, or hierarchical-risk-parity weights)
- runs factor / attribution regression to separate alpha from passive factor exposure
- generates multi-panel tear-sheet reports (equity, drawdown, rolling Sharpe, monthly heatmap, distribution, metrics table)
- tabulates periodic returns — a year x month table with an annual total, per-year returns, and rolling annualised return / volatility / Sharpe (`src.reporting.periodic`)
- exports trade logs and parameter sweep results
- ships an end-to-end `examples/full_demo.py` and a GitHub Actions CI workflow

## Project structure

```text
trading_system/
├── src/
│   ├── data/
│   │   ├── loader.py              # Yahoo Finance data download
│   │   ├── csv_loader.py          # Local CSV OHLCV loader (offline / custom data)
│   │   ├── cache.py               # Parquet-backed loader cache
│   │   ├── resample.py            # OHLCV bar aggregation (1m -> 1D, etc.)
│   │   ├── universe.py            # FAANG / Dow30 / sectors / benchmarks / factors
│   │   └── quality.py             # OHLCV data-quality auditor + cleaner
│   ├── indicators/                # Comprehensive vectorised TA library
│   │   ├── trend.py               # SMA / EMA / WMA / VWMA
│   │   ├── momentum.py            # RSI / MACD / Stochastic / Williams %R / CCI / ROC
│   │   ├── volatility.py          # ATR / Bollinger / Keltner / Donchian
│   │   └── volume.py              # OBV / VWAP (anchored) / Chaikin A/D
│   ├── strategy/
│   │   ├── base.py                # Abstract Strategy class + SmaCrossover example
│   │   ├── momentum.py            # Momentum signal generator
│   │   ├── mean_reversion.py      # Bollinger Bands + RSI strategy
│   │   ├── breakout.py            # Donchian channel breakout (turtle-style)
│   │   ├── ema_crossover.py       # Fast/slow EMA and MACD crossover signals
│   │   ├── trix.py                # TRIX triple-EMA trend-following strategy
│   │   ├── pairs.py               # Cointegration-based pairs trading
│   │   └── ensemble.py            # Majority / weighted / unanimous combiners
│   ├── backtest/
│   │   ├── engine.py              # Vectorised cost-aware backtest engine
│   │   └── event_engine.py        # Event-driven engine with full OMS
│   ├── oms/                       # Order Management System
│   │   ├── order.py               # Order / OrderStatus / OrderType / Side / TIF
│   │   ├── position.py            # Per-symbol position with cost basis + PnL
│   │   └── portfolio.py           # Cash + positions + equity history
│   ├── live/
│   │   └── broker.py              # Broker interface + PaperBroker implementation
│   ├── execution/
│   │   ├── slippage.py            # Spread + sqrt-impact execution model
│   │   └── impact.py              # Participation-rate cost + Almgren-Chriss scheduling
│   ├── risk/
│   │   ├── manager.py             # Risk management middleware
│   │   ├── sizing.py              # Kelly / ATR / fixed-fractional sizing
│   │   └── metrics.py             # VaR / CVaR / Omega / Ulcer / drawdown stats / rolling beta
│   ├── regime/
│   │   ├── detector.py            # Market regime detection (ADX + Hurst)
│   │   └── hmm.py                 # Gaussian HMM regime detector (Baum-Welch + Viterbi)
│   ├── validation/
│   │   ├── walk_forward.py        # Walk-forward validation framework
│   │   ├── monte_carlo.py         # Bootstrap + trade-shuffle robustness
│   │   ├── stat_tests.py          # Sharpe t-test, Probabilistic & Deflated SR
│   │   └── pbo.py                 # Probability of Backtest Overfitting (CSCV)
│   ├── portfolio/
│   │   ├── portfolio.py           # Multi-asset portfolio backtest
│   │   └── optimizer.py           # Min-variance / max-Sharpe / risk-parity weights
│   └── reporting/
│       ├── metrics.py             # Portfolio metrics + trade analytics
│       ├── plots.py               # Equity curve plotting
│       ├── trades.py              # Trade log builder (standalone utility)
│       ├── sweep.py               # Parameter sweep runner
│       ├── tearsheet.py           # Multi-panel PNG tear-sheet report
│       ├── attribution.py         # Factor / alpha regression
│       └── periodic.py            # Calendar return tables + rolling metrics
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
│   ├── test_csv_loader.py         # CSV loader tests
│   ├── test_ema_crossover.py      # EMA / MACD strategy tests
│   ├── test_pairs.py              # Pairs trading / cointegration tests
│   ├── test_ensemble.py           # Signal-ensemble tests
│   ├── test_stat_tests.py         # Sharpe significance / DSR tests
│   ├── test_oms.py                # OMS: Order / Position / Portfolio tests
│   ├── test_event_engine.py       # Event-driven engine tests
│   ├── test_strategy_base.py      # Strategy ABC + SmaCrossover tests
│   ├── test_indicators.py         # Indicators library tests
│   ├── test_risk_metrics.py       # VaR / CVaR / Omega / drawdown tests
│   ├── test_data_extras.py        # Cache / resample / universe tests
│   ├── test_live_broker.py        # PaperBroker tests
│   └── test_engine.py             # Backtest engine tests
├── examples/
│   └── full_demo.py               # End-to-end demo touching every module
├── .github/workflows/
│   └── ci.yml                     # CI: ruff + mypy + pytest/coverage (3.11 + 3.12)
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

### 2. Install the package

```bash
pip install -e .          # runtime install
pip install -e ".[dev]"   # add the lint / type / test toolchain
```

### 3. Run the project

```bash
# default: momentum strategy on SPY
python main.py

# mean reversion strategy on AAPL
python main.py --strategy mean-reversion --ticker AAPL

# adaptive (regime-based) strategy
python main.py --strategy adaptive

# Donchian breakout on QQQ
python main.py --strategy breakout --ticker QQQ --bo-entry 30 --bo-exit 10

# EMA crossover
python main.py --strategy ema-cross --ema-fast 20 --ema-slow 100

# MACD
python main.py --strategy macd

# load OHLCV from a local CSV instead of yfinance
python main.py --csv data/spy.csv

# realistic execution costs (spread + sqrt-impact) instead of flat --cost
python main.py --execution-model --spread-bps 5 --impact-coeff 0.1

# add Monte Carlo bootstrap on the daily return series
python main.py --monte-carlo 1000 --mc-block-size 5

# emit a multi-panel tearsheet PNG
python main.py --tearsheet

# Deflated Sharpe correcting for 100 parameter trials
python main.py --n-trials 100

# walk-forward validation
python main.py --walk-forward

# walk-forward with mean reversion and custom windows
python main.py --strategy mean-reversion --walk-forward --wf-is-days 252 --wf-oos-days 63

# disable risk management
python main.py --no-risk

# verbose logging
python main.py -v

# end-to-end demo of every module on synthetic data (no network needed)
python examples/full_demo.py
```

### 4. Run tests

```bash
pytest                                          # 323 tests
pytest --cov=src --cov-report=term-missing      # with coverage (floor: 80%, ~86% now)
```

### 5. Code quality

```bash
ruff check .          # lint
ruff format --check . # formatting
mypy src main.py grid_search.py plot_heatmap.py  # strict type checks

pre-commit install    # optional: run the checks automatically on each commit
```

All of the above run in CI (`.github/workflows/ci.yml`) on a Python 3.11 / 3.12
matrix and are expected to pass cleanly (ruff, ruff-format, mypy --strict, and
pytest + coverage).

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

### EMA / MACD crossover

Two classical trend-following baselines sharing the same EMA primitive. The EMA crossover compares a fast and slow EMA with an optional flat-zone gap to suppress signal churn. The MACD variant uses the standard (12, 26, 9) configuration of MACD line vs signal line.

```bash
python main.py --strategy ema-cross --ema-fast 20 --ema-slow 100 --ema-gap-bps 10
python main.py --strategy macd --macd-fast 12 --macd-slow 26 --macd-signal 9
```

### TRIX (triple-EMA trend)

TRIX is the rate of change of a triple-smoothed EMA — a low-noise trend oscillator. The strategy holds long while TRIX is positive and short while it is negative; `use_signal_line=True` switches the threshold to an EMA of TRIX for earlier crossover entries. Reuses the shared `trix` primitive from `src.indicators`.

```python
from src.strategy.trix import trix_strategy

signals = trix_strategy(df, period=15, use_signal_line=False, allow_short=True)
```

### Pairs trading (cointegration)

Stat-arb on two cointegrated price series. Engle-Granger test fits a hedge ratio by OLS, runs an Augmented Dickey-Fuller test on the residuals, and only allows trading when the residual is stationary at 5%. The spread is then z-scored on a rolling window and traded back to the mean.

```python
from src.strategy.pairs import pairs_trading_signal

signals = pairs_trading_signal(
    coca_cola_close, pepsi_close,
    z_window=60, z_entry=2.0, z_exit=0.5,
)
```

Returns a single-asset DataFrame compatible with the standard backtest engine (the spread is published as the `close` column).

### Signal ensemble

Combine multiple strategies into a single signal series in `src/strategy/ensemble.py`:

- **`majority_vote(signals)`** — sign of the row sum, ties → flat. Most robust to a misbehaving member.
- **`weighted_sum(signals, weights, threshold)`** — weighted average sign-thresholded. Useful when one strategy has higher conviction.
- **`unanimous(signals)`** — take a position only if every strategy agrees. Trades less often but with higher per-trade conviction.

All combiners take a wide-format DataFrame whose columns are individual `{-1, 0, +1}` signal series.

## Regime detection

The regime detector (`src/regime/detector.py`) classifies each day into one of three regimes using two independent indicators:

- **ADX** (Average Directional Index) measures trend strength regardless of direction. Values above 25 indicate a strong trend.
- **Hurst exponent** estimates persistence in the price series. H > 0.55 suggests trending behaviour; H < 0.45 suggests mean reversion; H near 0.5 is a random walk.

The final classification requires both indicators to agree. A rolling majority-vote smoothing window prevents whipsawing between regimes. Volatility regime (high/low) is also computed as a secondary signal.

### Hidden Markov model

For a probabilistic, data-driven alternative, `src/regime/hmm.py` fits a univariate Gaussian HMM to the return series with the Baum-Welch (EM) algorithm and decodes the most-likely regime path with Viterbi — pure numpy, no scipy. Fitting is deterministic (quantile initialisation) and states are relabelled by ascending mean, so a two-state model cleanly separates a calm, higher-mean regime from a turbulent, lower-mean one without label-switching.

```python
from src.regime import HMMConfig, detect_hmm_regime, fit_gaussian_hmm

states = detect_hmm_regime(returns, n_states=2)      # Series of 0/1 regime labels
result = fit_gaussian_hmm(returns.to_numpy(), HMMConfig(n_states=3))
print(result.state_means, result.transition, result.log_likelihood)
```

The scaled forward-backward pass keeps the likelihood numerically stable on long series, and `HMMResult.posterior` exposes the smoothed `P(state | data)` for soft-allocation use.

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

`src/risk/sizing.py` provides six independent sizing helpers that can be used at calibration time (fit on IS, apply on OOS) or as post-hoc studies:

- **`kelly_fraction(returns, cap, kelly_fraction_of_full)`** — continuous Kelly `f* = mean / variance`. Half-Kelly is the default to control variance.
- **`atr_position_size(price, atr, equity, risk_per_trade, atr_multiple, max_size)`** — sets notional so that hitting an `N * ATR` stop costs exactly `risk_per_trade` of equity.
- **`fixed_fractional(win_rate, payoff_ratio, cap)`** — discrete-outcome Kelly from trade-log stats: `f* = W - (1 - W) / R`.
- **`volatility_target_size(realized_vol, target_vol, max_size)`** — scale exposure so realised vol matches a target (`target / realized`).
- **`cppi_fraction(equity, floor, multiplier, max_size)`** — CPPI risky exposure on the cushion above a protective floor; de-risks automatically toward the floor.
- **`drawdown_throttle(current_drawdown, max_drawdown, max_size)`** — linearly cut exposure as the drawdown approaches a tolerated cap.

All helpers return a position fraction in `[0, cap]` and degrade gracefully (return `0.0`) when the edge is non-positive.

## Event-driven backtesting

The vectorised engine is fast, but it operates on signal series and can't model order types, partial fills, or intrabar dynamics. For execution-realistic simulation use the event-driven engine in `src.backtest.event_engine`:

```python
from src.backtest.event_engine import EventEngine
from src.strategy.base import SmaCrossoverStrategy

eng = EventEngine(
    symbol="SPY", initial_cash=100_000,
    commission_per_share=0.005, commission_min=1.0,
    slippage_bps=2.0,
)
result = eng.run(ohlcv_df, SmaCrossoverStrategy(fast=20, slow=50, trade_qty=100))

print(result.portfolio.cash)                # remaining cash
print(result.portfolio.positions["SPY"])    # cost basis + realized PnL
print(result.equity_curve.tail())
print(result.fills.head())
```

Supported order types:

| Type         | Fill rule                                                                                        |
|--------------|--------------------------------------------------------------------------------------------------|
| `MARKET`     | Next bar's open                                                                                  |
| `LIMIT`      | Fills if the bar's range crosses `limit_price` (gap-protected for sells / buys at open)         |
| `STOP`       | Triggers when the bar's high (buy) or low (sell) breaches `stop_price`; fills at worse of stop/open |
| `STOP_LIMIT` | Triggers like a STOP, then behaves like a LIMIT until expiry                                     |

Time-in-force `DAY`, `GTC`, `IOC`, `FOK` are modelled. Commissions and slippage are charged per fill; the OMS tracks weighted-avg cost basis per symbol and separates realised vs unrealised PnL.

### Writing event-driven strategies

Subclass `Strategy` and implement `on_bar(ctx)`:

```python
from src.strategy.base import Strategy
from src.oms import Side
from src.indicators import sma

class RsiMeanReversion(Strategy):
    def __init__(self, period=14, oversold=30, qty=10):
        self.period = period
        self.oversold = oversold
        self.qty = qty

    def on_bar(self, ctx):
        from src.indicators import rsi
        r = rsi(ctx.history["close"], self.period).iloc[-1]
        pos = ctx.portfolio.get_position(ctx.symbol)
        if r < self.oversold and pos.is_flat:
            ctx.submit_order(Side.BUY, self.qty)
        elif r > 50 and pos.is_long:
            ctx.submit_order(Side.SELL, pos.quantity)
```

### Paper broker

`src.live.broker.PaperBroker` shares the same OMS as the event engine, so paper-traded strategies port to a future live broker by swapping one constructor:

```python
from src.live.broker import PaperBroker
from src.oms import Order, Side, OrderType

bk = PaperBroker(initial_cash=100_000, commission_per_share=0.005)
bk.submit_order(Order(symbol="SPY", side=Side.BUY, quantity=100), mark_price=412.50)
bk.poll({"SPY": 415.10})        # re-evaluate working LIMIT/STOP orders
print(bk.equity({"SPY": 415.10}))
```

The roadmap for real-broker adapters (Interactive Brokers, Alpaca, Binance) is to subclass `Broker` and implement the same six methods.

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

### Optimal execution (Almgren-Chriss)

`src/execution/impact.py` adds the optimal-execution staples used to schedule a large order: `participation_rate_cost` (temporary impact as a power law of size / ADV — the square-root law at `exponent=0.5`) and the Almgren-Chriss liquidation schedule.

```python
from src.execution.impact import almgren_chriss_cost, almgren_chriss_trajectory

schedule = almgren_chriss_trajectory(total_shares=10_000, n_steps=20, urgency=0.7)
cost = almgren_chriss_cost(schedule, eta=0.1, gamma=0.001)
```

Higher `urgency` front-loads trading (less timing risk, more impact); `urgency=0` is plain TWAP, which minimises temporary impact for a given size.

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

## Sharpe significance & Deflated SR

`src/validation/stat_tests.py` adds three significance tests that go beyond a point-estimate Sharpe ratio:

- **`sharpe_ttest(returns)`** — classical Sharpe-ratio t-test under iid normal returns. Reports annualised Sharpe, t-stat and two-sided p-value.
- **`probabilistic_sharpe_ratio(returns, target_sharpe)`** — Bailey & López de Prado (2012) probability that the true Sharpe exceeds the target, with skew / excess-kurtosis corrections. Returns a value in `[0, 1]`.
- **`deflated_sharpe_ratio(returns, n_trials)`** — same idea but inflates the target Sharpe to account for the *best of N* parameter trials. The right answer to "I tried 200 parameter combinations and the best one looks great — is that real?" is almost never the raw PSR.

Both PSR and DSR are pure-Python (no scipy): they use the standard-library `math.erf` for the normal CDF and an Acklam-2003 rational approximation for its inverse.

## Probability of backtest overfitting (PBO)

When you grid-search hundreds of parameter combinations, the best in-sample config is often just the luckiest. `src/validation/pbo.py` quantifies that risk with **combinatorially-symmetric cross-validation** (Bailey, Borwein, López de Prado & Zhu, 2017):

```python
from src.validation import probability_of_backtest_overfitting

# returns_matrix: (T observations, N candidate configs)
result = probability_of_backtest_overfitting(returns_matrix, n_blocks=10)
print(result.pbo)   # ~0 generalises · ~0.5 no real edge · ~1 systematic overfit
```

The series is split into `n_blocks` equal blocks; every choice of half the blocks as in-sample (with the complement out-of-sample) is evaluated. For each split the in-sample-best config's out-of-sample rank is mapped to a logit, and PBO is the fraction of splits where it lands in the bottom half. Pure numpy.

## Multi-asset portfolio

`src/portfolio/portfolio.py` runs the same single-asset strategy across a basket of tickers and aggregates the per-asset return streams into a portfolio using one of three weighting schemes:

- **`equal`** — flat 1/N weights, rebalanced daily.
- **`inverse_vol`** — weights inversely proportional to trailing realised volatility (simple risk-parity proxy).
- **`custom`** — user-supplied static weights, normalised to sum to 1.

For covariance-aware allocation, `src/portfolio/optimizer.py` provides five closed-form / iterative schemes (no scipy required):

- **`min_variance_weights(returns, cov=None)`** — long-only minimum-variance portfolio via `Σ⁻¹ 1`.
- **`max_sharpe_weights(returns, cov=None, rf_daily=0)`** — long-only tangency portfolio via `Σ⁻¹ (μ − rf)`.
- **`risk_parity_weights(returns, cov=None)`** — true equal risk-contribution weights via cyclical coordinate descent (Maillard, Roncalli & Teïletche 2010).
- **`maximum_diversification_weights(returns, cov=None)`** — most-diversified portfolio via `Σ⁻¹ σ` (Choueifaty & Coignard 2008); reduces to inverse-vol for uncorrelated assets.
- **`hierarchical_risk_parity_weights(returns, cov=None)`** — HRP: correlation-distance clustering → quasi-diagonalisation → recursive bisection (López de Prado 2016).

The closed-form optimisers clip negative weights to zero and re-normalise; HRP is long-only by construction.

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

## Periodic return analytics

`src/reporting/periodic.py` tabulates a daily return stream over calendar periods — a useful companion to the tear-sheet:

```python
from src.reporting.periodic import annual_returns, monthly_returns_table, rolling_metrics

monthly = monthly_returns_table(returns)    # year x month, with an `annual` total column
yearly = annual_returns(returns)            # one compounded return per calendar year
roll = rolling_metrics(returns, window=63)  # rolling annualised return / volatility / Sharpe
```

All three expect a Series indexed by a DatetimeIndex and never mutate the input.

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

Core portfolio-level metrics computed from the daily return series:

- **Total Return** — cumulative return over the period
- **CAGR** — compound annual growth rate
- **Sharpe Ratio** — risk-adjusted return (annualised)
- **Sortino Ratio** — downside-risk-adjusted return
- **Max Drawdown** — largest peak-to-trough decline
- **Calmar Ratio** — CAGR divided by max drawdown

### Advanced risk metrics

`src/risk/metrics.py` adds a deep bench of professional-grade risk statistics:

- **`historical_var(returns, level)`** — empirical Value-at-Risk at the chosen confidence.
- **`historical_cvar(returns, level)`** — expected shortfall conditional on breaching VaR (always ≥ VaR).
- **`parametric_var(returns, level)`** — Gaussian VaR for sanity-checking against the historical estimate.
- **`omega_ratio(returns, target_return)`** — gains-above-target / losses-below-target.
- **`ulcer_index(returns)`** — RMS percentage drawdown, capturing both depth and persistence.
- **`gain_to_pain_ratio(returns)`** — daily-returns profit factor.
- **`drawdown_stats(returns)`** — depth, start/end/recovery dates, duration and recovery time in days.
- **`downside_deviation` / `upside_deviation`** — annualised one-sided volatility.
- **`tail_ratio(returns, level)`** — right-tail / left-tail magnitude ratio.
- **`common_ratio(returns)`** — CAGR / annualised vol (compound-return Sharpe variant).
- **`rolling_beta(strategy, benchmark, window)`** — time-varying beta vs a benchmark.
- **`skewness` / `kurtosis`** — distribution shape (excess kurtosis; > 0 = fat tails).
- **`tracking_error` / `information_ratio`** — annualised active risk vs a benchmark and the risk-adjusted out-performance per unit of it.
- **`sterling_ratio` / `burke_ratio`** — annualised return per unit of average drawdown / root-sum-square of drawdowns.

## Indicators library

`src/indicators` is the single source of truth for technical indicators across strategies and the event engine. All are pure pandas / numpy with consistent API:

| Family       | Indicators                                                                       |
|--------------|----------------------------------------------------------------------------------|
| Trend        | `sma`, `ema`, `wma`, `vwma`, `hma`, `aroon`                                      |
| Momentum     | `rsi`, `macd`, `stochastic`, `williams_r`, `cci`, `roc`, `trix`, `cmo`           |
| Volatility   | `atr` (sma/ema/wilder), `bollinger`, `keltner`, `donchian`                       |
| Volume       | `obv`, `vwap` (anchored), `chaikin_ad`, `mfi`                                    |

```python
from src.indicators import rsi, bollinger, atr, vwap

rsi14 = rsi(df["close"], period=14)
bb = bollinger(df["close"], window=20, num_std=2.0)
atr14 = atr(df["high"], df["low"], df["close"], period=14, smoothing="wilder")
vw = vwap(df["close"], df["volume"], anchor="D")
```

## Data layer

- **`load_yahoo_ohlcv` / `load_csv_ohlcv`** — primary downloaders.
- **`CachedLoader`** — drop-in wrapper that persists downloaded frames to `~/.trading_system_cache/` (parquet preferred, CSV fallback) so repeat backtests don't hit the network.
- **`resample_ohlcv` / `to_daily` / `to_weekly` / `to_monthly`** — aggregate intraday bars (open=first, high=max, low=min, close=last, volume=sum).
- **`get_universe(name)`** — pre-defined baskets: `faang`, `faang_plus`, `dow30`, `sectors`, `benchmarks`, `factors`.
- **`check_ohlcv(df)` / `clean_ohlcv(df)`** — audit a frame for duplicates, gaps, OHLC inconsistencies, extreme returns and stale prices (returns a `DataQualityReport`), then drop the untrustworthy rows.

```python
from src.data.quality import check_ohlcv, clean_ohlcv

report = check_ohlcv(df)
if not report.is_clean:
    print(report.issues)
    df = clean_ohlcv(df)
```

```python
from src.data import load_yahoo_ohlcv
from src.data.cache import CachedLoader
from src.data.universe import get_universe

loader = CachedLoader(load_yahoo_ohlcv)
basket = {t: loader(t, start="2015-01-01") for t in get_universe("sectors")}
```

## Parameter sweep

The parameter sweep runner tests a grid of lookback/threshold combinations and exports ranked results to CSV. Results now include trade-level stats alongside portfolio metrics.

```bash
python grid_search.py
```

Results are saved to `results/sweep_results.csv` and the top 10 configurations are printed to the console.

## Limitations

The codebase is research-grade — it ships the architecture and components of a professional trading stack, but is not a production execution venue. Current limitations:

- Default data source is Yahoo Finance (free, end-of-day quality).
- Regime detection offers both an ADX + Hurst indicator vote and a Gaussian HMM; richer models (non-Gaussian emissions, regime-switching GARCH) are future work.
- Execution-cost model is reduced-form: spread + sqrt-impact + commission. No order-book simulation, no queue position, no cross-venue routing.
- Portfolio optimisers clip negative weights and renormalise rather than solving a constrained QP (sufficient for research baskets but not for institutional sizing).
- Monte Carlo resampling assumes stationary returns (use block bootstrap for short-range autocorrelation).
- The paper broker fills synchronously at the supplied mark — real broker adapters (IB / Alpaca / Binance) are stubbed only at the interface level.
- Event engine is single-asset per run; multi-asset event-driven backtesting is left as future work.

## Tech stack

- Python 3.11+
- pandas, NumPy
- matplotlib, seaborn
- yfinance
- pytest (+ pytest-cov), ruff, mypy (strict, with pandas-stubs), pre-commit
- GitHub Actions CI

## Author

Built as a personal quant / systematic trading research project.
