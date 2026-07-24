# Systematic Trading Research Platform

A Python research platform for designing, testing, and evaluating systematic trading strategies on market data — modelled after professional quant-trading stacks.

This project is structured around three complementary backtesting modes:

1. **Vectorised mode** (`src.backtest.engine`) — signal-in / equity-out, blazing fast, ideal for parameter sweeps and walk-forward research.
2. **Event-driven mode** (`src.backtest.event_engine`) — bar-by-bar simulation with a full Order Management System (limit / stop / stop-limit orders, DAY/GTC/IOC/FOK, partial fills, commission + slippage, cost-basis position tracking). Same API surface as the paper-broker live adapter. A **signal bridge** (`src.backtest.signal_bridge`) replays any signal strategy through this engine, so the same idea is graded research-fast *and* execution-realistic.
3. **Weight-frame mode** (`src.backtest.weights`) — a vectorised multi-asset engine over target-weight frames (rotation strategies, portfolio optimisers), with turnover costs and its own walk-forward harness (`walk_forward_weights`).

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

**Data** (`src.data`):

- downloads OHLCV via `yfinance` or local CSV with a parquet-backed cache; pre-defined universes (FAANG, Dow 30, sectors, benchmarks, factors); bar resampling (1m → 1D → 1W → 1ME) and intraday session filtering (RTH windows, overnight sessions wrapping midnight)
- audits data quality (duplicates, OHLC inconsistencies, extreme returns, stale runs) and detects *missing* bars — modal frequency inference plus a gap report against an expected calendar grid
- trading-calendar helpers derived from the bar index itself (rebalance dates per W/M/Q/Y, trading day of month, turn-of-month logic); CRSP-style back-adjustment for splits and dividends
- return transforms pinned down once (simple/log/excess, returns↔prices round-trips), a wide price-panel builder for the multi-asset pipeline, and a synthetic GBM OHLCV generator for offline tests

**Signals & regimes** (`src.indicators`, `src.strategy`, `src.regime`):

- a 34-indicator vectorised TA library: trend (SMA/EMA/WMA/VWMA/HMA, Aroon, Vortex, Ichimoku, KAMA, Parabolic SAR, ADX/DI), momentum (RSI, MACD, Stochastic, StochRSI, Williams %R, CCI, ROC, TRIX, CMO, Elder-Ray, 12-1 skip-month momentum, 52-week-high distance), volatility (ATR, Bollinger, Keltner, Donchian, SuperTrend, Chaikin Volatility, Choppiness) and volume (OBV, anchored VWAP, Chaikin A/D, MFI)
- seventeen strategy templates: momentum, mean reversion, Donchian breakout, EMA crossover, MACD, TRIX, MFI, HMM regime-switching, squeeze, pairs (cointegration), adaptive (regime-based), time-series momentum (12-1 TSMOM), dual momentum (multi-asset rotation), turn-of-month seasonal, KAMA adaptive trend — plus majority/weighted/unanimous ensemble combiners
- regime detection four ways: ADX + Hurst voting, Gaussian HMM (Baum-Welch + Viterbi, plus a **causal forward filter** for out-of-sample use), trailing-quantile volatility regimes with hysteresis, and 20%-drawdown bull/bear labels; Markov analytics on top (transition matrix, dwell times, stationary distribution, k-step forecasts, entropy rate / predictability) and per-regime performance attribution

**Backtesting** (`src.backtest`):

- vectorised engine with costs, vol targeting and risk middleware; event-driven engine with the full OMS; weight-frame engine for rotation/allocation strategies; a signal→event-engine bridge so one strategy grades in both worlds
- walk-forward for weight rules (strictly out-of-sample refits), execution-lag sensitivity tables, per-trade MAE/MFE excursions, trade statistics, exposure/turnover summaries and equity/drawdown curves

**Execution & OMS** (`src.execution`, `src.oms`):

- cost models from reduced-form (spread + √-impact) to Almgren-Chriss trajectories; child-order schedules (TWAP, VWAP, POV, iceberg with size jitter); limit-fill simulation bracketed two ways (optimistic touch and FIFO queue burn-through); post-trade TCA (implementation shortfall, VWAP slippage, effective/realized spread + price impact decomposition); short-borrow and margin financing drag
- an OMS with strict order lifecycles, commission schedules, pre-trade risk checks (fat-finger caps, position/leverage limits, price collars, restricted lists), a Reg-T margin/buying-power model, weight→order rebalance planning, pro-rata block allocation, split/dividend bookkeeping and exposure/fill analytics; a `PaperBroker` sharing the same OMS

**Risk & portfolio** (`src.risk`, `src.portfolio`):

- the VaR family (historical, parametric, Cornish-Fisher modified, CVaR) plus factor-model VaR with systematic/idio decomposition and scenario stress testing (per-asset and factor-level shocks); 30+ performance ratios; seven position sizers; constant-volatility (Barroso-Santa-Clara) risk-managed scaling; stop/TP/trailing/daily-loss middleware
- portfolio construction: min-variance, max-Sharpe, risk parity, arbitrary risk budgets, max diversification, HRP, the Merton efficient frontier and Black-Litterman posteriors — fed by three covariance estimators (sample, Ledoit-Wolf shrinkage, EWMA/RiskMetrics) — with turnover-budgeted rebalancing and risk-contribution analytics

**Validation** (`src.validation`):

- walk-forward, Monte Carlo bootstrap and trade-shuffle robustness; Sharpe t-test, Probabilistic and Deflated Sharpe; PBO (CSCV); purged & embargoed K-fold; **CPCV** with backtest-path assembly down to per-path return series; White's Reality Check and **Hansen's SPA**; Treynor-Mazuy / Henriksson-Merton market-timing regressions

**Reporting** (`src.reporting`):

- factor/alpha attribution with capture ratios and rolling alpha/beta; benchmark head-to-head tables and a multi-strategy league table; calendar seasonality (month-of-year, day-of-week, turn-of-month); periodic return tables, drawdown episode tables, matplotlib tear-sheets and a self-contained HTML report

## Project structure

```text
trading_system/
├── src/                           # ~120 typed modules across 11 packages
│   ├── data/                      # loaders, cache, resample, sessions, universes,
│   │   │                          #   quality + gap detection, calendar, corporate
│   │   │                          #   actions, returns transforms, panel builder,
│   │   └── ...                    #   synthetic GBM
│   ├── indicators/                # 34-indicator vectorised TA library
│   │   ├── trend.py               #   SMA/EMA/WMA/VWMA/HMA, Aroon, Vortex, Ichimoku,
│   │   ├── momentum.py            #   KAMA, PSAR, ADX/DI · RSI, MACD, Stoch(+RSI),
│   │   ├── volatility.py          #   W%R, CCI, ROC, TRIX, CMO, Elder-Ray, 12-1 mom,
│   │   └── volume.py              #   52w-high · ATR, BB, KC, Donchian, SuperTrend,
│   │                              #   Chaikin vol, Choppiness · OBV, VWAP, A/D, MFI
│   ├── strategy/                  # 17 signal/weight templates + ensembles
│   │   └── ...                    #   (momentum family, mean reversion, breakouts,
│   │                              #   TSMOM, dual momentum, turn-of-month, KAMA...)
│   ├── backtest/
│   │   ├── engine.py              # vectorised signal engine
│   │   ├── event_engine.py        # event-driven engine with full OMS
│   │   ├── weights.py             # multi-asset weight-frame engine
│   │   ├── signal_bridge.py       # signal strategies -> event engine
│   │   ├── walk_forward_weights.py# strictly-OOS refits for weight rules
│   │   ├── robustness.py          # execution-lag sensitivity tables
│   │   ├── excursions.py          # per-trade MAE / MFE analysis
│   │   └── ...                    # curves, trades, exposure
│   ├── oms/                       # orders, positions, portfolio, fees, pre-trade
│   │   └── ...                    #   checks, Reg-T margin, rebalance planner,
│   │                              #   pro-rata allocation, corporate actions
│   ├── live/broker.py             # Broker interface + PaperBroker (same OMS)
│   ├── execution/                 # slippage, Almgren-Chriss, TWAP/VWAP/POV/iceberg,
│   │   └── ...                    #   limit fills + FIFO queue model, TCA, spreads,
│   │                              #   financing costs
│   ├── risk/                      # risk middleware, 7 sizers, VaR family (+CF,
│   │   └── ...                    #   factor-model), stress scenarios, BSC scaling,
│   │                              #   30+ ratios
│   ├── regime/                    # ADX+Hurst, HMM (+causal filter), vol regimes,
│   │   └── ...                    #   bull/bear labels, Markov analytics + entropy,
│   │                              #   turbulence, per-regime performance
│   ├── validation/                # walk-forward, MC, PSR/DSR, PBO, purged K-fold,
│   │   └── ...                    #   CPCV + path assembly, Reality Check, SPA,
│   │                              #   market-timing tests
│   ├── portfolio/                 # optimisers (MV/maxSharpe/RP/risk-budget/maxDiv/
│   │   └── ...                    #   HRP), frontier, Black-Litterman, Ledoit-Wolf,
│   │                              #   EWMA cov, turnover budgeting, analytics
│   └── reporting/                 # metrics, attribution (+rolling alpha/beta),
│       └── ...                    #   benchmark & league tables, seasonality,
│                                  #   periodic/drawdown tables, tearsheet, HTML report
├── tests/                         # 75+ test files, 1100+ tests, ~93.5% coverage
├── examples/full_demo.py          # end-to-end demo touching every module
├── .github/workflows/ci.yml       # CI: ruff + mypy --strict + pytest (3.11 + 3.12)
├── main.py                        # CLI entry point (single-asset signal pipeline)
├── grid_search.py                 # parameter sweep script
└── pyproject.toml                 # config: ruff, mypy strict, coverage gate
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
pytest                                          # 1100+ tests
pytest --cov=src --cov-report=term-missing      # with coverage (floor: 80%, ~93.5% now)
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

Seventeen templates ship in `src.strategy`. The classics below are documented in detail; the newer additions follow the same conventions — **TSMOM** (`tsmom.py`, Moskowitz-Ooi-Pedersen 12-1 time-series momentum with month-end decisions), **dual momentum** (`dual_momentum.py`, multi-asset relative+absolute rotation emitting a weight frame), **turn-of-month** (`turn_of_month.py`, the Lakonishok-Smidt −1/+3 seasonal window) and **KAMA trend** (`kama_trend.py`, adaptive-MA crossover with exit hysteresis). Any template's `signal` column runs through the vectorised engine, and via `run_signal_event_backtest` through the event engine too.

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
    entry_window=20,  # N-day high/low for entry
    exit_window=10,  # M-day opposite channel for exit
    atr_period=14,
    atr_filter=0.5,  # require breakout >= 0.5 * ATR
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

### MFI mean reversion

A volume-aware mean-reversion baseline: the Money Flow Index (a volume-weighted RSI) enters against extremes — long when MFI falls below `oversold`, short when it rises above `overbought` — and holds the position until MFI reverts to a neutral `exit_level`. Requires `high`, `low`, `close` and `volume`; reuses the shared `mfi` indicator.

```python
from src.strategy.mfi import mfi_strategy

signals = mfi_strategy(df, period=14, oversold=20, overbought=80, allow_short=True)
```

### HMM regime switching

Fits a Gaussian HMM to the return series and trades the latent regime: long in the highest-mean state, short in the lowest, flat in between (states sorted by mean). Reuses `src.regime.detect_hmm_regime`.

```python
from src.strategy.hmm_regime import hmm_regime_strategy

signals = hmm_regime_strategy(df, n_states=2, allow_short=True)
```

The HMM is fit in-sample, so run it inside walk-forward validation (which refits per fold) for leakage-controlled out-of-sample results.

### Squeeze breakout (Bollinger / Keltner)

Stays flat while the Bollinger Bands sit inside the Keltner Channels (a low-volatility squeeze) and takes the momentum direction once the squeeze releases. Needs `high`, `low` and `close`; reuses the shared `bollinger`, `keltner` and `sma` indicators.

```python
from src.strategy.squeeze import squeeze_strategy

signals = squeeze_strategy(df, bb_std=2.0, kc_atr_mult=1.5, allow_short=True)
```

### Pairs trading (cointegration)

Stat-arb on two cointegrated price series. Engle-Granger test fits a hedge ratio by OLS, runs an Augmented Dickey-Fuller test on the residuals, and only allows trading when the residual is stationary at 5%. The spread is then z-scored on a rolling window and traded back to the mean.

```python
from src.strategy.pairs import pairs_trading_signal

signals = pairs_trading_signal(
    coca_cola_close,
    pepsi_close,
    z_window=60,
    z_entry=2.0,
    z_exit=0.5,
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

states = detect_hmm_regime(returns, n_states=2)  # Series of 0/1 regime labels
result = fit_gaussian_hmm(returns.to_numpy(), HMMConfig(n_states=3))
print(result.state_means, result.transition, result.log_likelihood)
```

The scaled forward-backward pass keeps the likelihood numerically stable on long series, and `HMMResult.posterior` exposes the smoothed `P(state | data)` for soft-allocation use. Because the posterior conditions on the *whole* sample, live use goes through `src.regime.filter_hmm_probabilities` / `filtered_hmm_states` — a strictly causal forward filter over a fitted model (fit on a training window, filter new observations as they arrive).

### Regime transition analytics

`src/regime/transitions.py` summarises any regime-label series (from either detector or the HMM): `regime_transition_matrix` gives the empirical first-order Markov transition probabilities (each row sums to 1) and `regime_durations` gives the average dwell time per regime.

```python
from src.regime import regime_durations, regime_transition_matrix

states = detect_hmm_regime(returns, n_states=2)
transitions = regime_transition_matrix(states)  # P(next | current)
dwell = regime_durations(states)  # mean bars spent in each regime
```

## Risk management

Risk controls are applied as a middleware layer between position sizing and return calculation. Each rule can be independently enabled or disabled.

```python
from src.risk.manager import RiskConfig

risk_config = RiskConfig(
    stop_loss=0.05,  # exit if loss exceeds 5%
    take_profit=0.10,  # exit if profit reaches 10%
    trailing_stop=0.03,  # exit if 3% drawdown from peak
    max_position=1.0,  # cap position size at 100%
    daily_loss_limit=0.02,  # flatten all if daily loss exceeds 2%
)
```

CLI flags: `--stop-loss`, `--take-profit`, `--trailing-stop`, `--max-position`, `--daily-loss-limit`, `--no-risk`.

## Position sizing

`src/risk/sizing.py` provides seven independent sizing helpers that can be used at calibration time (fit on IS, apply on OOS) or as post-hoc studies:

- **`kelly_fraction(returns, cap, kelly_fraction_of_full)`** — continuous Kelly `f* = mean / variance`. Half-Kelly is the default to control variance.
- **`atr_position_size(price, atr, equity, risk_per_trade, atr_multiple, max_size)`** — sets notional so that hitting an `N * ATR` stop costs exactly `risk_per_trade` of equity.
- **`fixed_fractional(win_rate, payoff_ratio, cap)`** — discrete-outcome Kelly from trade-log stats: `f* = W - (1 - W) / R`.
- **`volatility_target_size(realized_vol, target_vol, max_size)`** — scale exposure so realised vol matches a target (`target / realized`).
- **`cppi_fraction(equity, floor, multiplier, max_size)`** — CPPI risky exposure on the cushion above a protective floor; de-risks automatically toward the floor.
- **`drawdown_throttle(current_drawdown, max_drawdown, max_size)`** — linearly cut exposure as the drawdown approaches a tolerated cap.
- **`optimal_f(trades, cap)`** — Ralph Vince optimal `f`: the fraction maximising geometric growth (Terminal Wealth Relative) over a set of trade outcomes.

All helpers return a position fraction in `[0, cap]` and degrade gracefully (return `0.0`) when the edge is non-positive.

## Event-driven backtesting

The vectorised engine is fast, but it operates on signal series and can't model order types, partial fills, or intrabar dynamics. For execution-realistic simulation use the event-driven engine in `src.backtest.event_engine`:

```python
from src.backtest.event_engine import EventEngine
from src.strategy.base import SmaCrossoverStrategy

eng = EventEngine(
    symbol="SPY",
    initial_cash=100_000,
    commission_per_share=0.005,
    commission_min=1.0,
    slippage_bps=2.0,
)
result = eng.run(ohlcv_df, SmaCrossoverStrategy(fast=20, slow=50, trade_qty=100))

print(result.portfolio.cash)  # remaining cash
print(result.portfolio.positions["SPY"])  # cost basis + realized PnL
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
bk.poll({"SPY": 415.10})  # re-evaluate working LIMIT/STOP orders
print(bk.equity({"SPY": 415.10}))
```

The roadmap for real-broker adapters (Interactive Brokers, Alpaca, Binance) is to subclass `Broker` and implement the same six methods.

### Portfolio exposure analytics

`src.oms.portfolio_exposure` turns a `Portfolio` + mark prices into an `ExposureReport` — gross / net / long / short exposure, leverage (gross / equity) and a Herfindahl concentration index (1 = a single position, → 0 = many equal positions):

```python
from src.oms import portfolio_exposure

rep = portfolio_exposure(portfolio, {"SPY": 410.0, "QQQ": 310.0})
print(rep.gross_exposure, rep.leverage, rep.concentration_hhi)
```

`summarize_fills(order.fills)` complements it at the order level, reporting the fill VWAP, the maker/taker quantity mix, and the time from first to last fill.

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

### Transaction-cost analysis (TCA)

`src/execution/tca.py` scores realised fills against the usual post-trade benchmarks — the arrival (decision) price and a market VWAP:

```python
from src.execution.tca import implementation_shortfall, vwap_slippage

is_cost = implementation_shortfall(fill_prices, fill_qty, arrival_price=412.5, side="buy")
slip = vwap_slippage(fill_prices, fill_qty, benchmark_vwap=413.1, side="buy")
```

Costs are signed fractions of the benchmark: positive = worse than the benchmark, negative = price improvement.

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
print(result.pbo)  # ~0 generalises · ~0.5 no real edge · ~1 systematic overfit
```

The series is split into `n_blocks` equal blocks; every choice of half the blocks as in-sample (with the complement out-of-sample) is evaluated. For each split the in-sample-best config's out-of-sample rank is mapped to a logit, and PBO is the fraction of splits where it lands in the bottom half. Pure numpy.

## Purged cross-validation

Plain K-fold leaks information in a backtest — a training bar adjacent to the test fold shares the same market state (and, for multi-bar labels, the same outcome). `src/validation/purged_cv.py` implements López de Prado's fix: **purge** training bars whose window overlaps the test fold and **embargo** a span immediately after it.

```python
from src.validation import purged_kfold_splits

for train_idx, test_idx in purged_kfold_splits(len(returns), n_splits=5, embargo=0.01, purge=5):
    ...  # leakage-safe train / test indices
```

## White's Reality Check

When you keep the best of many strategies, its edge is inflated by the search. `src/validation/reality_check.py` runs White's (2000) bootstrap reality check on a `(T, N)` panel of per-strategy performance, testing whether the *best* strategy beats the benchmark after correcting for having tried all N.

```python
from src.validation import whites_reality_check

result = whites_reality_check(excess_returns, n_bootstrap=1000, block_size=5, seed=0)
print(result.p_value, result.best_strategy)  # small p -> the best edge is real
```

A moving-block bootstrap preserves short-range autocorrelation; the p-value is the share of recentred bootstrap max-statistics that exceed the observed one.

`src/validation/spa.py` upgrades this to **Hansen's SPA test**: statistics are studentised (a noisy irrelevant candidate no longer dominates the max) and the consistent null keeps the negative means of significantly-bad strategies (stuffing the set with garbage no longer inflates the p-value). It returns the ordered `lower ≤ consistent ≤ upper` p-value triple. For path-level diagnostics, `src/validation/cpcv.py` generates combinatorial purged splits and reassembles per-split out-of-sample returns into the φ full backtest paths that PBO consumes.

## Multi-asset portfolio

`src/portfolio/portfolio.py` runs the same single-asset strategy across a basket of tickers and aggregates the per-asset return streams into a portfolio using one of three weighting schemes:

- **`equal`** — flat 1/N weights, rebalanced daily.
- **`inverse_vol`** — weights inversely proportional to trailing realised volatility (simple risk-parity proxy).
- **`custom`** — user-supplied static weights, normalised to sum to 1.

For covariance-aware allocation, `src/portfolio/optimizer.py` provides six closed-form / iterative schemes (no scipy required):

- **`min_variance_weights(returns, cov=None)`** — long-only minimum-variance portfolio via `Σ⁻¹ 1`.
- **`max_sharpe_weights(returns, cov=None, rf_daily=0)`** — long-only tangency portfolio via `Σ⁻¹ (μ − rf)`.
- **`risk_parity_weights(returns, cov=None)`** — true equal risk-contribution weights via cyclical coordinate descent (Maillard, Roncalli & Teïletche 2010).
- **`risk_budget_weights(returns, budgets, cov=None)`** — the arbitrary-budget generalisation: risk contributions proportional to any positive budget vector.
- **`maximum_diversification_weights(returns, cov=None)`** — most-diversified portfolio via `Σ⁻¹ σ` (Choueifaty & Coignard 2008); reduces to inverse-vol for uncorrelated assets.
- **`hierarchical_risk_parity_weights(returns, cov=None)`** — HRP: correlation-distance clustering → quasi-diagonalisation → recursive bisection (López de Prado 2016).

The closed-form optimisers clip negative weights to zero and re-normalise; HRP is long-only by construction.

Around the optimisers the package also ships:

- **`efficient_frontier(returns, n_points)`** — the Merton closed-form mean-variance frontier with min-vol / max-Sharpe locators.
- **`black_litterman(returns, market_weights, views, ...)`** — equilibrium-implied returns tilted toward investor views (He-Litterman posterior); the posterior covariance drops straight back into the optimisers.
- **`ledoit_wolf_covariance(returns)`** / **`ewma_covariance(returns, decay=0.94)`** — well-conditioned shrinkage and responsive RiskMetrics estimators as alternatives to the sample covariance (all optimisers accept `cov=`).
- **`constrain_turnover(current, target, max_turnover)`** / **`portfolio_turnover`** — turnover-budgeted rebalancing between any two weight vectors.
- **`walk_forward_weights(prices, weight_fn, ...)`** (in `src.backtest`) — rolling refits of any history→weights rule, traded strictly out of sample through the weight-frame engine, and **`rebalance_orders`** (in `src.oms`) to turn target weights into a broker-ready order list.

Risk diagnostics for a given weight vector + covariance live in `src/portfolio/analytics.py`:

```python
from src.portfolio import diversification_ratio, effective_number_of_assets, risk_contributions

rc = risk_contributions(weights, cov)  # each asset's share of portfolio variance (sums to 1)
dr = diversification_ratio(weights, cov)  # >= 1; higher = better diversified
n_eff = effective_number_of_assets(weights)  # inverse-Herfindahl "number of bets"
```

```python
from src.portfolio import PortfolioConfig, run_portfolio_backtest

basket = {"SPY": spy_df, "QQQ": qqq_df, "GLD": gld_df}
result = run_portfolio_backtest(
    basket,
    strategy_fn,
    backtest_fn,
    config=PortfolioConfig(weighting="inverse_vol", vol_window=20),
)
print(result.metrics)  # portfolio-level Sharpe / MaxDD / etc.
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

The same module exposes benchmark-relative **capture ratios**:

```python
from src.reporting.attribution import capture_ratio, down_capture, up_capture

up = up_capture(strategy_returns, benchmark_returns)  # > 1 amplifies upside
down = down_capture(strategy_returns, benchmark_returns)  # < 1 cushions downside
ratio = capture_ratio(strategy_returns, benchmark_returns)  # up / down; higher is better
```

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

monthly = monthly_returns_table(returns)  # year x month, with an `annual` total column
yearly = annual_returns(returns)  # one compounded return per calendar year
roll = rolling_metrics(returns, window=63)  # rolling annualised return / volatility / Sharpe
```

All three expect a Series indexed by a DatetimeIndex and never mutate the input.

### Drawdown table

`src/reporting/drawdowns.py` decomposes the equity path into distinct peak-to-trough-to-recovery episodes and returns the deepest ones:

```python
from src.reporting.drawdowns import drawdown_table

worst = drawdown_table(returns, top_n=5)  # peak / trough / recovery dates, depth, length
```

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
- **`treynor_ratio` / `jensen_alpha` / `m2_ratio`** — CAPM benchmark-relative performance: excess return per unit of beta, alpha vs the beta-predicted return, and the Modigliani M² (return re-levered to benchmark volatility).

## Indicators library

`src/indicators` is the single source of truth for technical indicators across strategies and the event engine. All are pure pandas / numpy with consistent API:

| Family       | Indicators                                                                       |
|--------------|----------------------------------------------------------------------------------|
| Trend        | `sma`, `ema`, `wma`, `vwma`, `hma`, `aroon`, `vortex`, `ichimoku`, `kama`, `parabolic_sar`, `adx` (+DI±) |
| Momentum     | `rsi`, `macd`, `stochastic`, `stoch_rsi`, `williams_r`, `cci`, `roc`, `trix`, `cmo`, `elder_ray`, `lookback_return` (12-1), `distance_from_high` (52w) |
| Volatility   | `atr` (sma/ema/wilder), `bollinger` (+%B/bandwidth), `keltner`, `donchian`, `supertrend`, `chaikin_volatility`, `choppiness` |
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
- **`get_universe(name)` / `list_universes()` / `combine_universes(...)` / `in_universe(ticker, name)`** — pre-defined baskets (`faang`, `faang_plus`, `dow30`, `sectors`, `benchmarks`, `factors`) with list / order-preserving-union / membership helpers.
- **`generate_gbm_ohlcv(...)`** — a geometric-Brownian-motion OHLCV generator for offline demos and tests; always OHLC-consistent and positive (passes `check_ohlcv`).
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
- Execution modelling is bar-level: reduced-form costs, bracketed limit-fill models (optimistic touch vs FIFO queue burn-through) and child-order schedules — but no full order-book simulation and no cross-venue routing.
- Portfolio optimisers clip negative weights and renormalise rather than solving a constrained QP (sufficient for research baskets but not for institutional sizing); CVaR optimisation would need an LP solver and is out of scope.
- Monte Carlo resampling assumes stationary returns (use block bootstrap for short-range autocorrelation).
- The paper broker fills synchronously at the supplied mark — real broker adapters (IB / Alpaca / Binance) are stubbed only at the interface level.
- The event engine is single-asset per run; multi-asset execution realism comes from the weight-frame engine's turnover-cost model instead.
- The `main.py` CLI covers the single-asset signal pipeline; the multi-asset weight pipeline (panel builder → optimiser/rotation → weight engine → rebalance orders) is currently library-only.

## Tech stack

- Python 3.11+
- pandas, NumPy
- matplotlib, seaborn
- yfinance
- pytest (+ pytest-cov), ruff, mypy (strict, with pandas-stubs), pre-commit
- GitHub Actions CI

## Author

Built as a personal quant / systematic trading research project.
