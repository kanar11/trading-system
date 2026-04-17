# Systematic Trading Research Backtester

A Python research project for designing, testing, and evaluating systematic trading strategies on market data.

This project demonstrates a full research workflow: data ingestion, signal generation, cost-aware backtesting with risk management, performance evaluation, trade-level analysis, parameter optimisation, and walk-forward validation.

## What the project does

The system currently:

- downloads historical OHLCV data using `yfinance` (any ticker)
- generates trading signals using three strategies: momentum, mean reversion, and adaptive (regime-based)
- detects market regimes (trending vs mean-reverting) using ADX and Hurst exponent
- applies a cost-aware backtest with transaction costs and volatility targeting
- enforces risk management rules (stop-loss, take-profit, trailing stop, position limits, daily loss limit)
- runs walk-forward validation to test strategy robustness (rolling IS/OOS)
- builds equity curves and saves them as plots
- calculates performance statistics (Sharpe, Sortino, Calmar, CAGR, Max Drawdown)
- exports trade logs and parameter sweep results

## Current features

- Yahoo Finance OHLCV data loader with validation
- **Momentum strategy** with configurable lookback, threshold, and SMA-200 regime filter
- **Mean reversion strategy** based on Bollinger Bands with RSI confirmation filter
- **Adaptive strategy** with automatic regime detection:
  - ADX (Average Directional Index) for trend strength
  - Rolling Hurst exponent for mean-reversion detection
  - Volatility regime classification (high/low)
  - Smoothing to prevent regime whipsawing
  - Auto-selects momentum in trends, mean reversion in ranges, flat in undefined
- Cost-aware backtesting engine with volatility targeting
- Risk management module:
  - stop-loss (configurable threshold)
  - take-profit (configurable threshold)
  - trailing stop (peak-to-trough)
  - maximum position size cap
  - daily loss limit (circuit breaker)
- **Walk-forward validation**:
  - rolling in-sample / out-of-sample windows
  - per-fold and aggregated OOS metrics
  - IS vs OOS Sharpe degradation analysis
  - combined OOS equity curve
- Performance metrics: Total Return, CAGR, Sharpe, Sortino, Max Drawdown, Calmar
- Trade log with entry/exit dates, prices, direction, return, and holding period
- Parameter sweep for comparing multiple strategy configurations
- Full CLI with argparse for reproducible research runs

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
│       ├── metrics.py             # Performance metrics
│       ├── plots.py               # Equity curve plotting
│       ├── trades.py              # Trade log builder (standalone)
│       └── sweep.py               # Parameter sweep runner
├── tests/
│   ├── conftest.py
│   ├── test_metrics.py
│   ├── test_risk_manager.py
│   ├── test_momentum.py
│   ├── test_mean_reversion.py
│   ├── test_walk_forward.py
│   ├── test_regime.py
│   └── test_engine.py
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

# custom momentum parameters
python main.py --lookback 100 --threshold 0.01 --no-sma-filter

# disable risk management
python main.py --no-risk

# walk-forward validation
python main.py --walk-forward

# adaptive (regime-based) strategy
python main.py --strategy adaptive

# walk-forward with mean reversion
python main.py --strategy mean-reversion --walk-forward --wf-is-days 252 --wf-oos-days 63

# verbose logging
python main.py -v
```

### 4. Run tests

```bash
pytest tests/ -v
```

## Strategies

### Momentum

Generates signals based on lookback returns with an optional SMA-200 regime filter. Goes long when the lookback return exceeds the threshold, short when it falls below.

```bash
python main.py --strategy momentum --lookback 200 --threshold 0.005
```

### Mean Reversion

Uses Bollinger Bands to detect overbought/oversold conditions. Enters when price moves outside the bands and exits when it reverts to the middle band. RSI acts as a confirmation filter.

```bash
python main.py --strategy mean-reversion --bb-window 20 --bb-std 2.0 --rsi-period 14
```

### Adaptive (Regime-Based)

Automatically detects the market regime using ADX (trend strength) and Hurst exponent (mean-reversion tendency). Applies the momentum strategy during trending periods, mean reversion during range-bound periods, and goes flat when the regime is unclear.

```bash
python main.py --strategy adaptive --adx-threshold 25 --hurst-window 100
```

## Walk-forward validation

Tests strategy robustness by splitting data into rolling in-sample (training) and out-of-sample (testing) windows. Reports per-fold metrics, combined OOS equity, and IS-vs-OOS Sharpe degradation.

```bash
python main.py --walk-forward --wf-is-days 504 --wf-oos-days 126
```

A degradation above 50% suggests the strategy may be overfitting to in-sample data.

## Risk management configuration

Risk controls are configured via CLI flags or through `RiskConfig`:

```python
risk_config = RiskConfig(
    stop_loss=0.05,        # exit if loss exceeds 5%
    take_profit=0.10,      # exit if profit reaches 10%
    trailing_stop=0.03,    # exit if 3% drawdown from peak
    max_position=1.0,      # cap position size at 100%
    daily_loss_limit=0.02, # flatten all if daily loss exceeds 2%
)
```

Set any parameter to `None` (or use `--no-risk` in CLI) to disable.

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
