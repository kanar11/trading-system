# Systematic Trading Research Backtester

A Python research project for designing, testing, and evaluating systematic trading strategies on market data.

This project demonstrates a full research workflow: data ingestion, signal generation, cost-aware backtesting with risk management, performance evaluation, trade-level analytics, parameter optimisation, and walk-forward validation.

## What the project does

The system currently:

- downloads historical OHLCV data using `yfinance` (any ticker)
- generates trading signals using three strategies: momentum, mean reversion, and adaptive (regime-based)
- detects market regimes (trending vs mean-reverting) using ADX and Hurst exponent
- applies a cost-aware backtest with transaction costs and volatility targeting
- enforces risk management rules (stop-loss, take-profit, trailing stop, position limits, daily loss limit)
- runs walk-forward validation to test strategy robustness (rolling IS/OOS)
- computes trade-level analytics (win rate, profit factor, expectancy, streaks, payoff ratio)
- builds equity curves and saves them as plots
- calculates performance statistics (Sharpe, Sortino, Calmar, CAGR, Max Drawdown)
- exports trade logs and parameter sweep results

## Project structure

```text
trading_system/
├── quantbt/                       # Importable package (pip install -e .)
│   ├── config.py                 # Validated RunConfig (pydantic + YAML + env)
│   ├── cli.py                    # CLI pipeline (config-driven)
│   ├── data/
│   │   └── loader.py              # Yahoo Finance data download
│   ├── strategy/
│   │   ├── momentum.py            # Momentum signal generator
│   │   ├── mean_reversion.py      # Bollinger Bands + RSI strategy
│   │   └── registry.py            # Strategy registry (@register / build_strategy)
│   ├── backtest/
│   │   └── engine.py              # Cost-aware backtest engine
│   ├── risk/
│   │   └── manager.py             # Risk management controls
│   ├── regime/
│   │   └── detector.py            # Market regime detection (ADX + Hurst)
│   ├── validation/
│   │   └── walk_forward.py        # Walk-forward validation framework
│   └── reporting/
│       ├── metrics.py             # Portfolio metrics + trade analytics
│       ├── plots.py               # Equity curve plotting
│       ├── trades.py              # Trade log builder (standalone utility)
│       └── sweep.py               # Parameter sweep runner
├── configs/
│   └── example.yaml               # Sample run configuration
├── tests/                         # Pytest suite (117 tests, ~86% coverage)
├── .github/workflows/ci.yml       # CI: ruff + mypy + pytest (py3.11/3.12)
├── .pre-commit-config.yaml        # Pre-commit hooks (ruff, mypy, hygiene)
├── main.py                        # Thin entry-point shim → quantbt.cli:main
├── grid_search.py                 # Grid search script
├── plot_heatmap.py                # Heatmap visualisation
├── pyproject.toml                 # Project config, dependencies, tooling
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
# editable install (exposes the `quantbt-backtest` console script)
pip install -e .
```

### 3. Run the project

The pipeline is config-driven. A run is fully described by a `RunConfig`
(`quantbt/config.py`): defaults < `QUANTBT_*` env vars < YAML file < CLI flags.
The resolved config is written to `results/run_config.yaml` for reproducibility.

```bash
# default: momentum strategy on SPY (python main.py and quantbt-backtest are equivalent)
quantbt-backtest

# run entirely from a YAML config
quantbt-backtest --config configs/example.yaml

# YAML base with a CLI override (flags win)
quantbt-backtest --config configs/example.yaml --ticker AAPL

# mean reversion strategy on AAPL
quantbt-backtest --strategy mean-reversion --ticker AAPL

# adaptive (regime-based) strategy
quantbt-backtest --strategy adaptive

# custom momentum parameters
quantbt-backtest --lookback 100 --threshold 0.01 --no-sma-filter

# walk-forward validation
quantbt-backtest --walk-forward --strategy mean-reversion --wf-is-days 252 --wf-oos-days 63

# override via environment variable
QUANTBT_DATA__TICKER=QQQ quantbt-backtest

# disable risk management / verbose logging
quantbt-backtest --no-risk -v
```

### 4. Run tests

```bash
pytest                                  # 117 tests
pytest --cov=quantbt --cov-report=term-missing   # with coverage (floor: 85%)
```

### 5. Code quality checks

Install the dev tooling and run the full check suite (lint, formatting, types):

```bash
pip install -e ".[dev]"

ruff check .          # lint
ruff format --check . # formatting
mypy quantbt main.py grid_search.py plot_heatmap.py tests  # strict type checks

# optional: install git hooks so checks run automatically on commit
pre-commit install
```

All checks (pytest + coverage, ruff lint, ruff format, mypy strict) run in CI
(`.github/workflows/ci.yml`) on a Python 3.11/3.12 matrix and are expected to
pass cleanly.

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

## Walk-forward validation

Tests strategy robustness by splitting data into rolling in-sample (training) and out-of-sample (testing) windows. For each fold, the strategy runs on the full IS+OOS window to avoid warm-up artefacts, but only the OOS portion is evaluated. Reports per-fold metrics, combined OOS equity, and IS-vs-OOS Sharpe degradation.

```bash
python main.py --walk-forward --wf-is-days 504 --wf-oos-days 126
```

A Sharpe degradation above 50% suggests the strategy may be overfitting to in-sample data. Below 20% indicates good robustness.

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

This is a research prototype, not a production trading system. Current limitations include single instrument per run, Yahoo Finance data only, no portfolio-level construction, and simplified regime detection (not a hidden Markov model).

## Tech stack

- Python 3.11+
- pandas, NumPy
- matplotlib, seaborn
- yfinance
- pydantic, pydantic-settings, PyYAML (typed config)
- pytest (+ pytest-cov), ruff, mypy (with pandas-stubs), pre-commit
- GitHub Actions CI

## Author

Built as a personal quant / systematic trading research project.
