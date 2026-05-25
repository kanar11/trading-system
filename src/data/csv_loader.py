"""Local CSV loader for OHLCV data.

A drop-in replacement for ``src.data.loader.load_yahoo_ohlcv`` that
reads from a local CSV file instead of hitting Yahoo Finance. Useful
for:

    * offline / air-gapped runs,
    * deterministic CI tests,
    * exotic instruments not on Yahoo (custom futures, crypto, etc.),
    * pre-downloaded snapshots so backtest runs are reproducible.

The loader is forgiving about column names — it normalises the
common variants (``Open``/``open``/``OPEN``, ``Adj Close`` → ``close``
when requested, ``Date``/``Datetime``/``timestamp`` as the index).
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

REQUIRED_COLS = ["open", "high", "low", "close", "volume"]

_DATE_CANDIDATES = ("date", "datetime", "timestamp", "time")
_COL_ALIASES = {
    "adj close": "adj_close",
    "adj_close": "adj_close",
    "adjclose": "adj_close",
}


def load_csv_ohlcv(
    path: str | Path,
    date_col: str | None = None,
    use_adj_close: bool = False,
    start: str | None = None,
    end: str | None = None,
) -> pd.DataFrame:
    """Load OHLCV data from a CSV file.

    The CSV must contain at least open / high / low / close / volume
    columns (case-insensitive, common spellings tolerated) and a date
    column. If ``date_col`` is omitted the loader auto-detects from a
    short list of common names.

    Args:
        path: Path to the CSV file.
        date_col: Explicit name of the date column. If None, auto-detect.
        use_adj_close: If True and the CSV has an "Adj Close" column,
            substitute it for the regular close (handles dividends /
            splits in Yahoo CSV exports).
        start: Optional inclusive start date (YYYY-MM-DD or ISO).
        end: Optional inclusive end date.

    Returns:
        DataFrame indexed by date with lowercase OHLCV columns.

    Raises:
        FileNotFoundError: If the CSV does not exist.
        ValueError: If required columns are missing or the file is empty.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"CSV not found: {p}")

    df = pd.read_csv(p)
    if df.empty:
        raise ValueError(f"CSV is empty: {p}")

    # normalise column names
    df.columns = [
        _COL_ALIASES.get(c.strip().lower(), c.strip().lower()) for c in df.columns
    ]

    # date column
    if date_col is None:
        for cand in _DATE_CANDIDATES:
            if cand in df.columns:
                date_col = cand
                break
    else:
        date_col = date_col.strip().lower()

    if date_col is None or date_col not in df.columns:
        raise ValueError(
            f"CSV must contain a date column. Tried: {list(_DATE_CANDIDATES)} "
            f"and explicit date_col={date_col!r}. Got columns: {list(df.columns)}"
        )

    df[date_col] = pd.to_datetime(df[date_col])
    df = df.set_index(date_col).sort_index()

    if use_adj_close and "adj_close" in df.columns:
        logger.info("Substituting adj_close for close (use_adj_close=True).")
        df["close"] = df["adj_close"]

    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(
            f"CSV is missing required columns: {missing}. "
            f"Available: {list(df.columns)}"
        )

    df = df[REQUIRED_COLS].dropna()

    if start is not None:
        df = df[df.index >= pd.to_datetime(start)]
    if end is not None:
        df = df[df.index <= pd.to_datetime(end)]

    if df.empty:
        raise ValueError(
            f"No rows left after applying start={start!r} / end={end!r} filters."
        )

    logger.info("Loaded %d rows from %s.", len(df), p)
    return df
