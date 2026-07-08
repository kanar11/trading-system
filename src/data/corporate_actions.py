"""Corporate-action back-adjustment (splits and cash dividends).

Raw price series jump on corporate actions: a 2:1 split halves the price
overnight and an ex-dividend date drops it by roughly the dividend — neither
is a real return, and any indicator or backtest run on raw prices will trade
those phantom moves. The industry fix is *back-adjustment* (what Yahoo's
"Adj Close" and CRSP do): scale all bars **before** each event by a
multiplicative factor so the series is return-continuous, while the latest
prices remain at their traded levels.

Factors: a split with ratio ``r`` (2.0 = 2-for-1) contributes ``1/r`` to
every bar before its effective date; a cash dividend ``d`` going ex on date
``t`` contributes ``1 − d / close_prev`` (total-return adjustment against
the last close before ``t``). Volume is scaled by splits only (multiplied by
``r`` before the split), never by dividends. Events dated on or before the
first bar are no-ops. Direct-import module::

    from src.data.corporate_actions import adjust_ohlcv, adjustment_factors
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

_PRICE_COLUMNS = ("open", "high", "low", "close")


def _validate_close(close: pd.Series) -> None:
    """Require a positive, NaN-free close on a sorted, unique DatetimeIndex."""
    if not isinstance(close.index, pd.DatetimeIndex):
        raise TypeError(f"close must have a DatetimeIndex, got {type(close.index).__name__}.")
    if not close.index.is_monotonic_increasing:
        raise ValueError("close index must be sorted in increasing order.")
    if close.index.has_duplicates:
        raise ValueError("close index must not contain duplicate timestamps.")
    values = close.to_numpy(dtype=float)
    if np.isnan(values).any() or (values <= 0).any():
        raise ValueError("close prices must be positive and NaN-free.")


def adjustment_factors(
    close: pd.Series,
    splits: Mapping[pd.Timestamp | str, float] | None = None,
    dividends: Mapping[pd.Timestamp | str, float] | None = None,
) -> pd.Series:
    """Multiplicative back-adjustment factor per bar (latest bars = 1.0).

    Args:
        close: Raw close prices on a sorted, unique DatetimeIndex.
        splits: ``{effective_date: ratio}``; ratio 2.0 = 2-for-1 split,
            0.5 = 1-for-2 reverse split.
        dividends: ``{ex_date: cash_amount}`` per share.

    Returns:
        Series named ``"adj_factor"`` aligned to ``close``; multiply raw
        prices by it to get the back-adjusted series.

    Raises:
        TypeError: If the index is not a DatetimeIndex.
        ValueError: If prices are invalid, a split ratio is not positive,
            or a dividend is negative or >= the preceding close.
    """
    _validate_close(close)
    factor = np.ones(len(close))
    index = close.index

    for key, ratio in (splits or {}).items():
        if ratio <= 0:
            raise ValueError(f"split ratio for {key!r} must be > 0, got {ratio}.")
        factor[index < pd.Timestamp(key)] /= ratio

    for key, amount in (dividends or {}).items():
        if amount < 0:
            raise ValueError(f"dividend for {key!r} must be >= 0, got {amount}.")
        before = index < pd.Timestamp(key)
        if not before.any():
            continue  # no bars precede the ex-date -> nothing to adjust
        prev_close = float(close.to_numpy()[before.nonzero()[0][-1]])
        if amount >= prev_close:
            raise ValueError(
                f"dividend {amount} on {key!r} is >= the preceding close {prev_close}."
            )
        factor[before] *= 1.0 - amount / prev_close

    return pd.Series(factor, index=index, name="adj_factor")


def adjust_ohlcv(
    df: pd.DataFrame,
    splits: Mapping[pd.Timestamp | str, float] | None = None,
    dividends: Mapping[pd.Timestamp | str, float] | None = None,
) -> pd.DataFrame:
    """Back-adjust an OHLCV frame for splits and dividends.

    Price columns (any of ``open``/``high``/``low``/``close`` present) are
    multiplied by the combined factor; ``volume``, when present, is scaled
    by the *split-only* factor (shares multiply in a split, but dividends
    leave volume untouched).

    Args:
        df: OHLCV frame with at least a ``close`` column on a sorted,
            unique DatetimeIndex.
        splits: ``{effective_date: ratio}`` (see :func:`adjustment_factors`).
        dividends: ``{ex_date: cash_amount}`` per share.

    Returns:
        A new, adjusted frame (input is not mutated).

    Raises:
        TypeError: If the index is not a DatetimeIndex.
        ValueError: If ``close`` is missing or the events are invalid.
    """
    if "close" not in df.columns:
        raise ValueError("DataFrame must contain a 'close' column.")

    combined = adjustment_factors(df["close"], splits=splits, dividends=dividends)
    out = df.copy()
    for column in _PRICE_COLUMNS:
        if column in out.columns:
            out[column] = out[column] * combined

    if "volume" in out.columns:
        split_only = adjustment_factors(df["close"], splits=splits)
        out["volume"] = out["volume"] / split_only

    return out
