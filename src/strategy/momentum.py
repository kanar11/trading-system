import pandas as pd


def momentum(df, lookback=20, threshold=0.01, use_sma_filter=False):
    df = df.copy()

    # momentum signal
    df["returns"] = df["close"].pct_change(lookback)

    df["signal"] = 0
    df.loc[df["returns"] > threshold, "signal"] = 1
    df.loc[df["returns"] < -threshold, "signal"] = -1

    # optional SMA200 regime filter
    if use_sma_filter:
        df["sma200"] = df["close"].rolling(200).mean()

        long_condition = (df["signal"] == 1) & (df["close"] > df["sma200"])
        short_condition = (df["signal"] == -1) & (df["close"] < df["sma200"])

        df["signal"] = 0
        df.loc[long_condition, "signal"] = 1
        df.loc[short_condition, "signal"] = -1

    return df


momentum_strategy = momentum