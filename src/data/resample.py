"""OHLCV bar aggregation.

Aggregate a high-frequency OHLCV frame (e.g. 1-minute bars) into
lower-frequency bars (5m, 15m, 1h, 1d, 1w). Respects the per-field
aggregation rule: open=first, high=max, low=min, close=last,
volume=sum.
"""

from __future__ import annotations

import pandas as pd

_OHLCV_AGG = {
    "open": "first",
    "high": "max",
    "low": "min",
    "close": "last",
    "volume": "sum",
}


def resample_ohlcv(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    """Resample OHLCV bars to a coarser frequency.

    Args:
        df: DataFrame with at least one of open/high/low/close/volume
            columns and a DatetimeIndex.
        rule: pandas offset alias — e.g. ``"5min"``, ``"15min"``,
            ``"1h"``, ``"1D"``, ``"1W"``.

    Returns:
        Resampled DataFrame with the same column ordering as the input.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame must have a DatetimeIndex.")

    cols = [c for c in df.columns if c in _OHLCV_AGG]
    if not cols:
        raise ValueError(
            "DataFrame must contain at least one of open/high/low/close/volume."
        )

    agg_spec = {c: _OHLCV_AGG[c] for c in cols}
    out = df[cols].resample(rule).agg(agg_spec).dropna(how="all")
    return out


def to_daily(df: pd.DataFrame) -> pd.DataFrame:
    """Convenience: resample to daily bars."""
    return resample_ohlcv(df, "1D")


def to_weekly(df: pd.DataFrame) -> pd.DataFrame:
    """Convenience: resample to weekly (Sunday-end) bars."""
    return resample_ohlcv(df, "1W")


def to_monthly(df: pd.DataFrame) -> pd.DataFrame:
    """Convenience: resample to month-end bars."""
    return resample_ohlcv(df, "1ME")
