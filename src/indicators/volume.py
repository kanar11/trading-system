"""Volume-based indicators."""

from __future__ import annotations

import numpy as np
import pandas as pd


def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    """On-Balance Volume.

    Adds volume on up-days, subtracts on down-days, leaves unchanged on
    unchanged-close days. Cumulative.
    """
    sign: pd.Series = np.sign(close.diff()).fillna(0)
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

    groups = pd.DatetimeIndex(price.index).to_period(anchor)
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


def mfi(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    volume: pd.Series,
    period: int = 14,
) -> pd.Series:
    """Money Flow Index — a volume-weighted RSI, in [0, 100].

    Splits each bar's raw money flow (typical price * volume) into positive /
    negative buckets by the change in typical price, then forms the RSI-style
    ratio over ``period`` bars. Saturates at 100 when there is no negative flow.
    """
    if period <= 0:
        raise ValueError("period must be > 0")
    typical = (high + low + close) / 3
    raw_flow = typical * volume
    direction = typical.diff()
    positive = raw_flow.where(direction > 0, 0.0)
    negative = raw_flow.where(direction < 0, 0.0)
    pos_sum = positive.rolling(period, min_periods=period).sum()
    neg_sum = negative.rolling(period, min_periods=period).sum()
    money_ratio = pos_sum / neg_sum.replace(0.0, np.nan)
    result: pd.Series = 100 - 100 / (1 + money_ratio)
    # positive flow but zero negative flow -> MFI saturates at 100
    return result.mask((neg_sum == 0) & (pos_sum > 0), 100.0)
