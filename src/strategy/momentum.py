import pandas as pd

def momentum(df, lookback=20, threshold=0.01):
    df = df.copy()

    df["returns"] = df["close"].pct_change(lookback)

    df["signal"] = 0
    df.loc[df["returns"] > threshold, "signal"] = 1
    df.loc[df["returns"] < -threshold, "signal"] = -1

    return df


momentum_strategy = momentum