"""Market data loading utilities.

Downloads OHLCV data from Yahoo Finance and normalises column names.
"""

import pandas as pd
import yfinance as yf

REQUIRED_COLS = ["open", "high", "low", "close", "volume"]


def load_yahoo_ohlcv(ticker: str, start: str = "2015-01-01") -> pd.DataFrame:
    """Download daily OHLCV data from Yahoo Finance.

    Args:
        ticker: Yahoo Finance symbol (e.g. 'SPY', 'AAPL').
        start: Start date in 'YYYY-MM-DD' format.

    Returns:
        DataFrame indexed by date with lowercase OHLCV columns.

    Raises:
        ValueError: If downloaded data is empty or missing required columns.
    """
    df = yf.download(ticker, start=start, progress=False, auto_adjust=True)

    if df.empty:
        raise ValueError(f"No data returned for ticker '{ticker}' from {start}.")

    # flatten MultiIndex columns (yfinance sometimes returns these)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]

    df = df.rename(
        columns={
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
        }
    )

    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns after download: {missing}")

    df = df[REQUIRED_COLS].dropna()
    return df
