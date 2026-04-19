"""Momentum-based trading signal generator.

Computes a lookback return and generates long/short signals
with an optional SMA-200 regime filter.
"""

import logging

import pandas as pd

logger = logging.getLogger(__name__)


def momentum_strategy(
    df: pd.DataFrame,
    lookback: int = 20,
    threshold: float = 0.01,
    use_sma_filter: bool = False,
) -> pd.DataFrame:
    """Generate momentum trading signals.

    Signal logic:
        +1 (long)  when lookback return > threshold
        -1 (short) when lookback return < -threshold
         0 (flat)  otherwise

    If ``use_sma_filter`` is True, long signals are only kept when
    price is above SMA-200 and short signals when price is below.

    Args:
        df: DataFrame with at least a 'close' column.
        lookback: Number of periods for return calculation.
        threshold: Minimum absolute return to trigger a signal.
        use_sma_filter: Whether to apply the SMA-200 regime filter.

    Returns:
        DataFrame with an added 'signal' column.
    """
    df = df.copy()

    df["returns"] = df["close"].pct_change(lookback)

    df["signal"] = 0
    df.loc[df["returns"] > threshold, "signal"] = 1
    df.loc[df["returns"] < -threshold, "signal"] = -1

    if use_sma_filter:
        df["sma200"] = df["close"].rolling(200).mean()

        long_ok = (df["signal"] == 1) & (df["close"] > df["sma200"])
        short_ok = (df["signal"] == -1) & (df["close"] < df["sma200"])

        df["signal"] = 0
        df.loc[long_ok, "signal"] = 1
        df.loc[short_ok, "signal"] = -1

    return df
