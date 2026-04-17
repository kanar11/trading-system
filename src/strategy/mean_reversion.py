"""Mean reversion strategy based on Bollinger Bands and RSI.

Generates trading signals when price deviates significantly from its
moving average, expecting a reversion to the mean. Uses Bollinger Bands
for entry/exit zones and RSI as a confirmation filter.
"""

import numpy as np
import pandas as pd


def _bollinger_bands(
    series: pd.Series,
    window: int = 20,
    num_std: float = 2.0,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Calculate Bollinger Bands.

    Args:
        series: Price series (typically close).
        window: Rolling window for the moving average.
        num_std: Number of standard deviations for the bands.

    Returns:
        Tuple of (middle_band, upper_band, lower_band).
    """
    middle = series.rolling(window).mean()
    std = series.rolling(window).std()
    upper = middle + num_std * std
    lower = middle - num_std * std
    return middle, upper, lower


def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Calculate Relative Strength Index.

    Args:
        series: Price series (typically close).
        period: Lookback period for RSI calculation.

    Returns:
        RSI values between 0 and 100.
    """
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)

    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def mean_reversion_strategy(
    df: pd.DataFrame,
    bb_window: int = 20,
    bb_std: float = 2.0,
    rsi_period: int = 14,
    rsi_oversold: float = 30.0,
    rsi_overbought: float = 70.0,
    use_rsi_filter: bool = True,
) -> pd.DataFrame:
    """Generate mean reversion trading signals.

    Signal logic:
        +1 (long)  when price closes below the lower Bollinger Band
                   (and RSI < oversold threshold if filter is enabled)
        -1 (short) when price closes above the upper Bollinger Band
                   (and RSI > overbought threshold if filter is enabled)
         0 (flat)  when price returns inside the bands (exit signal)

    The strategy enters positions at extremes and exits when price
    reverts back to the middle band.

    Args:
        df: DataFrame with at least a 'close' column.
        bb_window: Bollinger Bands moving average window.
        bb_std: Number of standard deviations for the bands.
        rsi_period: RSI calculation period.
        rsi_oversold: RSI level below which asset is considered oversold.
        rsi_overbought: RSI level above which asset is considered overbought.
        use_rsi_filter: Whether to require RSI confirmation for entries.

    Returns:
        DataFrame with added indicator and 'signal' columns.
    """
    df = df.copy()

    # Bollinger Bands
    df["bb_middle"], df["bb_upper"], df["bb_lower"] = _bollinger_bands(
        df["close"], window=bb_window, num_std=bb_std
    )

    # RSI
    df["rsi"] = _rsi(df["close"], period=rsi_period)

    # %B indicator — where price sits relative to the bands (0 = lower, 1 = upper)
    band_width = df["bb_upper"] - df["bb_lower"]
    df["percent_b"] = np.where(
        band_width > 0,
        (df["close"] - df["bb_lower"]) / band_width,
        0.5,
    )

    # generate raw signals
    df["signal"] = 0

    # long: price below lower band
    long_condition = df["close"] < df["bb_lower"]
    # short: price above upper band
    short_condition = df["close"] > df["bb_upper"]

    if use_rsi_filter:
        long_condition = long_condition & (df["rsi"] < rsi_oversold)
        short_condition = short_condition & (df["rsi"] > rsi_overbought)

    df.loc[long_condition, "signal"] = 1
    df.loc[short_condition, "signal"] = -1

    # hold position until price crosses back to middle band
    # (forward-fill signals, exit when crossing middle)
    position = 0
    signals = df["signal"].values.copy()
    close = df["close"].values
    middle = df["bb_middle"].values

    for i in range(len(df)):
        if signals[i] != 0:
            # new entry signal
            position = signals[i]
        elif position != 0:
            # check if price has reverted to middle band
            if position == 1 and close[i] >= middle[i]:
                position = 0  # exit long
            elif position == -1 and close[i] <= middle[i]:
                position = 0  # exit short
        signals[i] = position

    df["signal"] = signals

    return df
