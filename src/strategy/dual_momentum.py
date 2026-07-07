"""Dual momentum asset-rotation strategy.

Antonacci's dual momentum (2014) combines two filters over a basket of
assets: **relative momentum** ranks the assets by lookback return and picks
the strongest, while **absolute momentum** vetoes any pick whose own
lookback return is not above a threshold (classically the T-bill return, 0
by default) — failed slots sit in cash instead. Decisions are made on
periodic rebalance dates (last trading day of each month by default, via
:mod:`src.data.calendar`) and held until the next one.

Unlike the single-asset generators in this package, the input is a *wide*
close-price frame (columns = assets) and the output is a frame of target
weights in ``[0, 1]`` per asset, rows summing to at most 1 (the remainder is
cash). Weights are decided on the rebalance bar's close; shifting them for
next-bar execution is the backtest engine's job, as with ``signal`` columns.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src.data.calendar import rebalance_mask

logger = logging.getLogger(__name__)


def dual_momentum_strategy(
    prices: pd.DataFrame,
    lookback: int = 252,
    top_n: int = 1,
    absolute_threshold: float = 0.0,
    rebalance: str = "M",
) -> pd.DataFrame:
    """Generate dual-momentum rotation weights over an asset basket.

    On each rebalance date (last bar of the ``rebalance`` period): rank
    assets by their ``lookback`` return, take the ``top_n`` strongest
    (relative momentum), then zero out any winner whose return is not above
    ``absolute_threshold`` (absolute momentum) — its slot stays in cash.
    Selected assets get equal weight ``1 / top_n``; weights are held
    unchanged between rebalance dates. Before the first rebalance with a
    valid momentum reading, all weights are 0 (cash).

    Args:
        prices: Wide DataFrame of close prices (columns = assets) on a
            sorted, duplicate-free DatetimeIndex.
        lookback: Momentum lookback in bars (e.g. 252 = 12 months of days).
        top_n: Number of relative-momentum winners to hold.
        absolute_threshold: Minimum lookback return a winner must beat to be
            held (0.0 = must be positive; use the T-bill return for GEM).
        rebalance: Rebalance period — ``"W"``, ``"M"``, ``"Q"`` or ``"Y"``.

    Returns:
        DataFrame of target weights, same index/columns as ``prices``; each
        row sums to at most 1.

    Raises:
        ValueError: If ``prices`` has no columns, or ``lookback`` / ``top_n``
            is < 1 (index/frequency problems raise from the calendar
            helpers).
    """
    if prices.shape[1] == 0:
        raise ValueError("prices must have at least one column.")
    if lookback < 1:
        raise ValueError(f"lookback must be >= 1, got {lookback}.")
    if top_n < 1:
        raise ValueError(f"top_n must be >= 1, got {top_n}.")

    index = pd.DatetimeIndex(prices.index)
    mask = rebalance_mask(index, freq=rebalance, which="last")
    momentum = prices.pct_change(lookback, fill_method=None)

    weights = pd.DataFrame(np.nan, index=prices.index, columns=prices.columns)
    slot_weight = 1.0 / top_n
    for pos in np.flatnonzero(mask.to_numpy()):
        readings = momentum.iloc[int(pos)].dropna()
        row = pd.Series(0.0, index=prices.columns)
        if not readings.empty:
            winners = readings.nlargest(top_n)  # relative momentum
            held = winners[winners > absolute_threshold]  # absolute veto
            row.loc[held.index] = slot_weight
        weights.iloc[int(pos)] = row

    # hold each decision until the next rebalance; all-cash before the first
    return weights.ffill().fillna(0.0)
