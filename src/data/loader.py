import pandas as pd
import yfinance as yf

REQUIRED_COLS = ["open" , "high" , "low" , "close" , "volume"]

def load_yahoo_ohlcv(ticker: str, start="2015-01-01"):
    df = yf.download(ticker, start=start, progress=False)

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]

    df = df.rename(columns={
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Volume": "volume"
    })

    df = df[REQUIRED_COLS].dropna()
    return df



