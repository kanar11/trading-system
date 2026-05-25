"""Vectorised technical-indicator library (single source of truth).

Until now indicators were scattered across individual strategy files
(SMA in ``momentum.py``, RSI / Bollinger in ``mean_reversion.py``,
ADX in ``regime/detector.py``, ATR in ``strategy/breakout.py``,
EMA in ``strategy/ema_crossover.py`` and ``executor.py`` calculators).

This package re-exports the same primitives under one roof so:

    * Strategies can share consistent implementations.
    * The event-driven engine and live trader can compute indicators
      cheaply on rolling windows.
    * Indicators get one unified test suite.

All indicators take pandas Series / DataFrames and return Series.
NumPy is used internally where it helps; the public surface is pure
pandas. Functions are pure (no in-place mutation of inputs).
"""

from src.indicators.trend import sma, ema, wma, vwma
from src.indicators.momentum import rsi, macd, stochastic, williams_r, cci, roc
from src.indicators.volatility import atr, bollinger, keltner, donchian
from src.indicators.volume import obv, vwap, chaikin_ad

__all__ = [
    # trend
    "sma", "ema", "wma", "vwma",
    # momentum
    "rsi", "macd", "stochastic", "williams_r", "cci", "roc",
    # volatility
    "atr", "bollinger", "keltner", "donchian",
    # volume
    "obv", "vwap", "chaikin_ad",
]
