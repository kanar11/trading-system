"""Bollinger/Keltner squeeze breakout strategy.

A volatility-contraction / expansion play in the spirit of Carter's TTM
Squeeze: when the Bollinger Bands sit *inside* the Keltner Channels the market
is in a low-volatility "squeeze" and the strategy stays flat. When the squeeze
releases (bands expand back outside the channels) it takes a position in the
direction of momentum (price vs its moving average).

Reuses the shared ``bollinger`` / ``keltner`` / ``sma`` primitives from
:mod:`src.indicators`.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src.indicators import bollinger, keltner, sma

logger = logging.getLogger(__name__)

_REQUIRED = ("high", "low", "close")


def squeeze_strategy(
    df: pd.DataFrame,
    bb_window: int = 20,
    bb_std: float = 2.0,
    kc_window: int = 20,
    kc_atr_period: int = 10,
    kc_atr_mult: float = 1.5,
    momentum_window: int = 20,
    allow_short: bool = True,
) -> pd.DataFrame:
    """Bollinger/Keltner squeeze breakout signals.

    Signal logic:
        squeeze ON  (Bollinger inside Keltner) -> flat
        squeeze OFF + price above its MA       -> +1 (long)
        squeeze OFF + price below its MA       -> -1 (short, if allowed)

    Args:
        df: DataFrame with 'high', 'low' and 'close' columns.
        bb_window: Bollinger moving-average window.
        bb_std: Bollinger standard deviations.
        kc_window: Keltner EMA window.
        kc_atr_period: ATR period for the Keltner channels.
        kc_atr_mult: ATR multiple for the Keltner channel width.
        momentum_window: SMA window for the momentum (price vs MA) direction.
        allow_short: If False, short signals are clamped to flat.

    Returns:
        DataFrame with ``squeeze_on`` and ``signal`` columns.

    Raises:
        ValueError: If a required column is missing.
    """
    missing = [c for c in _REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"DataFrame must contain columns {missing}.")

    df = df.copy()
    bands = bollinger(df["close"], window=bb_window, num_std=bb_std)
    channels = keltner(
        df["high"],
        df["low"],
        df["close"],
        window=kc_window,
        atr_period=kc_atr_period,
        atr_mult=kc_atr_mult,
    )

    # comparisons involving warm-up NaNs evaluate False, so squeeze_on is a
    # plain boolean series (False = not squeezed / unknown during warm-up)
    squeeze_on = (bands["lower"] > channels["lower"]) & (bands["upper"] < channels["upper"])
    df["squeeze_on"] = squeeze_on

    momentum = df["close"] - sma(df["close"], momentum_window)
    released = ~squeeze_on

    signal = np.zeros(len(df), dtype=int)
    signal[(released & (momentum > 0)).to_numpy()] = 1
    if allow_short:
        signal[(released & (momentum < 0)).to_numpy()] = -1

    df["signal"] = signal
    return df
