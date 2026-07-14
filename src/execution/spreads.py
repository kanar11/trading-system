"""Trade-cost decomposition: quoted, effective and realized spreads.

The microstructure TCA standard (SEC Rule 605 vocabulary) decomposes what
a trade actually paid versus the quote midpoint at arrival:

    effective spread  = 2 · side · (trade_price − mid)
    realized  spread  = 2 · side · (trade_price − mid_future)
    price impact      = 2 · side · (mid_future  − mid)

with ``side`` +1 for buys and −1 for sells, and ``mid_future`` the
midpoint some minutes after the trade. The identity
``effective = realized + impact`` splits the paid spread into what the
liquidity provider kept (realized) and what was adverse selection — the
market moving with the trade (impact). ``quoted_spread`` is the resting
``ask − bid`` for context.

All functions are vectorised over aligned Series and return per-trade
series in price units (divide by the mid for relative versions), so they
compose with :mod:`src.execution.tca`'s order-level metrics.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _side_values(side: pd.Series | int, index: pd.Index) -> pd.Series:
    """Validate a trade-side input (+1 buy / -1 sell) and return a Series."""
    if isinstance(side, pd.Series):
        if not side.index.equals(index):
            raise ValueError("side must share the index of the price inputs.")
        values = side.to_numpy(dtype=float)
    else:
        values = np.full(len(index), float(side))
    if not np.isin(values, (-1.0, 1.0)).all():
        raise ValueError("side values must be +1 (buy) or -1 (sell).")
    return pd.Series(values, index=index)


def _aligned_positive(a: pd.Series, b: pd.Series, names: str) -> None:
    if not a.index.equals(b.index):
        raise ValueError(f"{names} must share the same index.")
    stacked = np.concatenate([a.to_numpy(dtype=float), b.to_numpy(dtype=float)])
    if np.isnan(stacked).any() or (stacked <= 0).any():
        raise ValueError(f"{names} must be positive and NaN-free.")


def quoted_spread(bid: pd.Series, ask: pd.Series, relative: bool = False) -> pd.Series:
    """Resting quoted spread ``ask − bid`` (or relative to the midpoint).

    Raises:
        ValueError: If the series are misaligned, non-positive, or any
            ask is below its bid.
    """
    _aligned_positive(bid, ask, "bid/ask")
    if (ask < bid).any():
        raise ValueError("ask must be >= bid on every row.")
    spread = ask - bid
    if relative:
        spread = spread / ((ask + bid) / 2.0)
    return spread.rename("quoted_spread")


def effective_spread(
    trade_price: pd.Series,
    mid: pd.Series,
    side: pd.Series | int,
) -> pd.Series:
    """Effective spread paid: ``2 · side · (trade_price − mid)``.

    Positive = the trade crossed toward the far side of the quote;
    negative = price improvement beyond the midpoint.

    Raises:
        ValueError: If inputs are misaligned/non-positive or ``side`` is
            not ±1.
    """
    _aligned_positive(trade_price, mid, "trade_price/mid")
    sides = _side_values(side, trade_price.index)
    return ((trade_price - mid) * sides * 2.0).rename("effective_spread")


def realized_spread(
    trade_price: pd.Series,
    future_mid: pd.Series,
    side: pd.Series | int,
) -> pd.Series:
    """Realized spread: ``2 · side · (trade_price − future_mid)``.

    What the liquidity provider actually earned once the market settled —
    the effective spread net of adverse selection.

    Raises:
        ValueError: If inputs are misaligned/non-positive or ``side`` is
            not ±1.
    """
    _aligned_positive(trade_price, future_mid, "trade_price/future_mid")
    sides = _side_values(side, trade_price.index)
    return ((trade_price - future_mid) * sides * 2.0).rename("realized_spread")


def price_impact(
    mid: pd.Series,
    future_mid: pd.Series,
    side: pd.Series | int,
) -> pd.Series:
    """Adverse-selection component: ``2 · side · (future_mid − mid)``.

    Positive when the midpoint moves in the trade's direction — the part
    of the effective spread the market took back. By construction
    ``effective_spread = realized_spread + price_impact``.

    Raises:
        ValueError: If inputs are misaligned/non-positive or ``side`` is
            not ±1.
    """
    _aligned_positive(mid, future_mid, "mid/future_mid")
    sides = _side_values(side, mid.index)
    return ((future_mid - mid) * sides * 2.0).rename("price_impact")
