# Systematic Trading Research Backtester

A Python-based research project for testing systematic momentum trading strategies on SPY market data.

The project demonstrates a complete research workflow:
- market data ingestion
- signal generation
- cost-aware backtesting
- performance evaluation
- trade-level analytics
- parameter sweep experimentation


## Features

- Yahoo Finance OHLCV data loader
- Momentum strategy with configurable lookback and threshold
- Backtesting engine with transaction costs
- Equity curve generation
- Strategy metrics:
  - Total Return
  - CAGR
  - Sharpe Ratio
  - Max Drawdown
- Trade log with entry/exit dates and holding periods
- Parameter sweep for comparing multiple strategy configurations

## Project Structure

```
trading_system/
│
├── src/
│   ├── data/
│   │   └── loader.py
│   │
│   ├── strategy/
│   │   └── momentum.py
│   │
│   ├── backtest/
│   │   └── engine.py
│   │
│   └── reporting/
│       ├── metrics.py
│       ├── plots.py
│       ├── trades.py
│       └── sweep.py
│
├── main.py
├── requirements.txt
└── README.md
```
## How to Run

Create a virtual environment and install the required dependencies:

```bash
pip install -r requirements.txt
