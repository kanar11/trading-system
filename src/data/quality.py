"""Data-quality auditing for OHLCV frames.

Institutional pipelines never trust raw vendor data: a single bad bar — a zero
price, a duplicated timestamp, a 10x fat-finger print, a stretch of stale
quotes — silently corrupts every downstream backtest. This module audits an
OHLCV frame, reports the problems it finds, and offers a conservative cleaner.

Pure pandas; the input is never mutated.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import pandas as pd

logger = logging.getLogger(__name__)

_PRICE_COLS = ("open", "high", "low", "close")


@dataclass
class DataQualityReport:
    """Summary of data-quality issues found in an OHLCV frame.

    Every count is 0 (and ``unsorted_index`` False) on clean data; ``issues``
    holds a human-readable line per problem and ``is_clean`` is then True.
    """

    n_rows: int
    duplicate_timestamps: int
    unsorted_index: bool
    missing_values: int
    non_positive_prices: int
    ohlc_inconsistencies: int
    extreme_returns: int
    max_stale_run: int
    issues: list[str] = field(default_factory=list)

    @property
    def is_clean(self) -> bool:
        """True when no quality issues were recorded."""
        return not self.issues


def _max_stale_run(close: pd.Series) -> int:
    """Longest run of consecutive identical closing prices (>= 1)."""
    c = close.dropna()
    if c.empty:
        return 0
    block = (c != c.shift()).cumsum()
    return int(c.groupby(block).size().max())


def check_ohlcv(
    df: pd.DataFrame,
    return_threshold: float = 0.5,
    max_stale: int = 5,
) -> DataQualityReport:
    """Audit an OHLCV frame and return a :class:`DataQualityReport`.

    Args:
        df: OHLCV DataFrame indexed by timestamp. Missing columns are simply
            skipped by the checks that need them.
        return_threshold: Absolute close-to-close return above which a bar is
            flagged as an extreme move (default 0.5 = 50%).
        max_stale: A run of identical closes longer than this is flagged.

    Returns:
        A populated :class:`DataQualityReport`.
    """
    issues: list[str] = []
    price_cols = [c for c in _PRICE_COLS if c in df.columns]
    value_cols = [c for c in (*_PRICE_COLS, "volume") if c in df.columns]

    duplicate_timestamps = int(df.index.duplicated().sum())
    unsorted_index = not bool(df.index.is_monotonic_increasing)
    missing_values = int(df[value_cols].isna().sum().sum()) if value_cols else 0

    non_positive_prices = int((df[price_cols] <= 0).any(axis=1).sum()) if price_cols else 0

    ohlc_inconsistencies = 0
    if all(c in df.columns for c in _PRICE_COLS):
        bad = (
            (df["high"] < df["low"])
            | (df["high"] < df["open"])
            | (df["high"] < df["close"])
            | (df["low"] > df["open"])
            | (df["low"] > df["close"])
        )
        ohlc_inconsistencies = int(bad.sum())

    extreme_returns = 0
    max_stale = max(max_stale, 1)
    max_stale_run = 0
    if "close" in df.columns:
        returns = df["close"].pct_change()
        extreme_returns = int((returns.abs() > return_threshold).sum())
        max_stale_run = _max_stale_run(df["close"])

    if duplicate_timestamps:
        issues.append(f"{duplicate_timestamps} duplicate timestamp(s)")
    if unsorted_index:
        issues.append("index is not sorted ascending")
    if missing_values:
        issues.append(f"{missing_values} missing value(s) in OHLCV columns")
    if non_positive_prices:
        issues.append(f"{non_positive_prices} row(s) with non-positive prices")
    if ohlc_inconsistencies:
        issues.append(f"{ohlc_inconsistencies} row(s) with OHLC inconsistencies")
    if extreme_returns:
        issues.append(f"{extreme_returns} extreme return(s) (>|{return_threshold:.0%}|)")
    if max_stale_run > max_stale:
        issues.append(f"stale price run of {max_stale_run} bars (> {max_stale})")

    return DataQualityReport(
        n_rows=len(df),
        duplicate_timestamps=duplicate_timestamps,
        unsorted_index=unsorted_index,
        missing_values=missing_values,
        non_positive_prices=non_positive_prices,
        ohlc_inconsistencies=ohlc_inconsistencies,
        extreme_returns=extreme_returns,
        max_stale_run=max_stale_run,
        issues=issues,
    )


def clean_ohlcv(df: pd.DataFrame, drop_non_positive: bool = True) -> pd.DataFrame:
    """Return a conservatively cleaned copy of an OHLCV frame.

    Sorts the index, drops duplicate timestamps (keeping the last), and removes
    rows with missing or (optionally) non-positive prices. Does not interpolate
    or forward-fill — it only removes untrustworthy rows.

    Args:
        df: OHLCV DataFrame.
        drop_non_positive: Drop rows where any price column is <= 0.

    Returns:
        A new, cleaned DataFrame.
    """
    out = df.copy()
    out = out.sort_index()
    out = out[~out.index.duplicated(keep="last")]

    price_cols = [c for c in _PRICE_COLS if c in out.columns]
    if price_cols:
        out = out.dropna(subset=price_cols)
        if drop_non_positive:
            out = out[(out[price_cols] > 0).all(axis=1)]
    return out
