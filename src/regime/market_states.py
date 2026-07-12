"""Bull / bear market state labels (drawdown-threshold definition).

The industry convention — used by index providers and in the momentum
literature (e.g. Cooper, Gutierrez & Hameed 2004 condition momentum on
market state; Daniel & Moskowitz 2016 date momentum crashes to bear-market
rebounds) — defines a **bear market** as a decline of at least ``threshold``
(classically 20%) from the running peak, and a return to a **bull market**
as a rally of the same magnitude off the subsequent trough.

The labelling is causal: the state at bar ``t`` uses only prices up to
``t`` (a running peak/trough state machine), so the labels can condition a
live strategy — e.g. only take TSMOM shorts in bear states — and feed
:func:`src.regime.conditional.regime_performance` directly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

BULL = 1
BEAR = -1


def bull_bear_labels(close: pd.Series, threshold: float = 0.20) -> pd.Series:
    """Label each bar +1 (bull) or -1 (bear) by the drawdown convention.

    Starting in a bull state at the first bar: a close at or below
    ``(1 - threshold) ×`` the running peak flips the state to bear from
    that bar; a close at or above ``(1 + threshold) ×`` the running trough
    flips it back to bull. Peaks reset on re-entry to bull, troughs on
    entry to bear.

    Args:
        close: Positive, NaN-free close prices.
        threshold: Fractional move defining a state change (0.20 = the
            classic 20% rule).

    Returns:
        Integer Series named ``"market_state"`` of +1 / -1 labels.

    Raises:
        ValueError: If ``threshold`` is outside (0, 1) or prices are
            non-positive / NaN.
    """
    if not 0.0 < threshold < 1.0:
        raise ValueError(f"threshold must be in (0, 1), got {threshold}.")
    prices = close.to_numpy(dtype=float)
    if len(prices) == 0:
        raise ValueError("close must not be empty.")
    if np.isnan(prices).any() or (prices <= 0).any():
        raise ValueError("close prices must be positive and NaN-free.")

    states = np.empty(len(prices), dtype=int)
    state = BULL
    peak = prices[0]
    trough = prices[0]
    for i, price in enumerate(prices):
        if state == BULL:
            peak = max(peak, price)
            if price <= peak * (1.0 - threshold):
                state = BEAR
                trough = price
        else:
            trough = min(trough, price)
            if price >= trough * (1.0 + threshold):
                state = BULL
                peak = price
        states[i] = state
    return pd.Series(states, index=close.index, name="market_state")
