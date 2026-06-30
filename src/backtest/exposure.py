"""Position-series analytics for vectorised backtest output.

Summarises the *holdings* path a strategy took — how often it was in the
market, its average long/short exposure, and how much it traded — straight from
the engine's position column. This is complementary to the return-based metrics
in :mod:`src.reporting` (which describe PnL) and to the point-in-time OMS
exposure report (which values open positions at marks).

Pure pandas; the input is never mutated.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class ExposureSummary:
    """Summary of a strategy's position (holdings) path.

    Attributes:
        time_in_market: Fraction of bars with a non-zero position.
        avg_exposure: Mean absolute position across all bars.
        avg_long_exposure: Mean position over bars that were long (0 if none).
        avg_short_exposure: Mean absolute position over short bars (0 if none).
        long_fraction: Fraction of bars that were long.
        short_fraction: Fraction of bars that were short.
        turnover: Sum of |position changes| (counting the initial entry from 0).
        n_trades: Number of bars where the position changed (entries, exits,
            flips), counting the first non-zero bar as an entry.
    """

    time_in_market: float
    avg_exposure: float
    avg_long_exposure: float
    avg_short_exposure: float
    long_fraction: float
    short_fraction: float
    turnover: float
    n_trades: int


def summarize_exposure(position: pd.Series) -> ExposureSummary:
    """Summarise a position/holdings series.

    Args:
        position: Per-bar position (e.g. the engine's ``position`` or
            ``scaled_position`` column). Sign encodes direction; magnitude
            encodes size.

    Returns:
        A populated :class:`ExposureSummary` (all-zero for an empty series).
    """
    p = pd.Series(position).dropna()
    if p.empty:
        return ExposureSummary(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0)

    longs = p > 0
    shorts = p < 0
    delta = p - p.shift(fill_value=0.0)

    return ExposureSummary(
        time_in_market=float((p != 0).mean()),
        avg_exposure=float(p.abs().mean()),
        avg_long_exposure=float(p[longs].mean()) if bool(longs.any()) else 0.0,
        avg_short_exposure=float(-p[shorts].mean()) if bool(shorts.any()) else 0.0,
        long_fraction=float(longs.mean()),
        short_fraction=float(shorts.mean()),
        turnover=float(delta.abs().sum()),
        n_trades=int((delta != 0).sum()),
    )
