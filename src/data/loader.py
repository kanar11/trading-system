"""Market data loading utilities.

Downloads OHLCV data from Yahoo Finance, normalises column names, and
validates the result before handing it to the rest of the pipeline.
"""

import logging

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

REQUIRED_COLS = ["open", "high", "low", "close", "volume"]

_COLUMN_RENAMES = {
    "Open": "open",
    "High": "high",
    "Low": "low",
    "Close": "close",
    "Adj Close": "close",
    "Volume": "volume",
}


def load_yahoo_ohlcv(
    ticker: str,
    start: str = "2015-01-01",
    end: str | None = None,
) -> pd.DataFrame:
    """Download daily OHLCV data from Yahoo Finance.

    Args:
        ticker: Yahoo Finance symbol (e.g. 'SPY', 'AAPL'). Case-insensitive.
        start: Start date in 'YYYY-MM-DD' format.
        end: Optional end date in 'YYYY-MM-DD' format. Defaults to today.

    Returns:
        DataFrame indexed by a sorted DatetimeIndex with lowercase OHLCV
        columns and no missing values.

    Raises:
        ValueError: If ``ticker`` is empty, the dates are malformed, the
            download is empty, or required columns are missing.
        RuntimeError: If the download itself fails (e.g. network error).
    """
    symbol = ticker.strip().upper()
    if not symbol:
        raise ValueError("Ticker must be a non-empty string.")

    start_ts = _parse_date(start, "start")
    end_ts = _parse_date(end, "end") if end is not None else None
    if end_ts is not None and end_ts <= start_ts:
        raise ValueError(f"end ({end}) must be after start ({start}).")

    logger.debug("Downloading %s from %s to %s", symbol, start, end or "today")
    try:
        raw = yf.download(
            symbol,
            start=start,
            end=end,
            progress=False,
            auto_adjust=True,
        )
    except Exception as exc:  # noqa: BLE001 - surface any yfinance/network failure uniformly
        raise RuntimeError(f"Failed to download data for '{symbol}': {exc}") from exc

    if raw is None or raw.empty:
        raise ValueError(f"No data returned for ticker '{symbol}' from {start}.")

    df = pd.DataFrame(raw)

    # yfinance sometimes returns a (field, ticker) MultiIndex on the columns.
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.rename(columns=_COLUMN_RENAMES)
    df = df.loc[:, ~df.columns.duplicated()]

    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns after download: {missing}")

    df = df[REQUIRED_COLS].dropna()
    if df.empty:
        raise ValueError(f"All rows for '{symbol}' were dropped as incomplete.")

    df.index = pd.DatetimeIndex(df.index)
    df = df.sort_index()
    return df


def _parse_date(value: str, label: str) -> pd.Timestamp:
    """Parse a 'YYYY-MM-DD' date string, raising a clear error on failure."""
    try:
        return pd.Timestamp(value)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"Invalid {label} date '{value}': {exc}") from exc
