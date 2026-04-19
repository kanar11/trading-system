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
├── src/
│   ├── data/
│   │   └── loader.py              # Yahoo Finance data download
│   ├── strategy/
│   │   ├── momentum.py            # Momentum signal generator
│   │   └── mean_reversion.py      # Bollinger Bands + RSI strategy
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
├── tests/
│   ├── conftest.py                # Shared fixtures (OHLCV, returns)
│   ├── test_metrics.py            # Portfolio + trade-level metric tests
│   ├── test_risk_manager.py       # Risk controls tests
│   ├── test_momentum.py           # Momentum strategy tests
│   ├── test_mean_reversion.py     # Mean reversion strategy tests
│   ├── test_walk_forward.py       # Walk-forward validation tests
│   ├── test_regime.py             # Regime detection tests
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
- pytest

## Author

Built as a personal quant / systematic trading research project.
