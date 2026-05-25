"""Volume-based indicators."""

from __future__ import annotations

import numpy as np
import pandas as pd


def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    """On-Balance Volume.

    Adds volume on up-days, subtracts on down-days, leaves unchanged on
    unchanged-close days. Cumulative.
    """
    sign = np.sign(close.diff()).fillna(0)
    return (sign * volume).cumsum().rename("obv")


def vwap(
    price: pd.Series,
    volume: pd.Series,
    anchor: str | None = "D",
) -> pd.Series:
    """Anchored Volume-Weighted Average Price.

    Args:
        price: Typical price series (commonly ``(high + low + close) / 3``).
        volume: Volume series.
        anchor: pandas offset alias to reset the anchor at
            (``"D"`` = daily, ``"W"`` = weekly, ``"ME"`` = month-end,
            ``None`` = cumulative-since-start).

    Returns:
        Series of running VWAP within each anchor bucket.
    """
    pv = price * volume
    if anchor is None:
        return pv.cumsum() / volume.cumsum()

    groups = price.index.to_period(anchor)
    cum_pv = pv.groupby(groups).cumsum()
    cum_v = volume.groupby(groups).cumsum()
    return (cum_pv / cum_v).rename("vwap")


def chaikin_ad(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    volume: pd.Series,
) -> pd.Series:
    """Chaikin Accumulation/Distribution line.

    AD_t = AD_{t-1} + ((close - low) - (high - close)) / (high - low) * volume.
    Falls back to 0 multiplier when high == low on a bar.
    """
    rng = (high - low).replace(0, np.nan)
    mfm = ((close - low) - (high - close)) / rng
    mfm = mfm.fillna(0)
    mfv = mfm * volume
    return mfv.cumsum().rename("chaikin_ad")
