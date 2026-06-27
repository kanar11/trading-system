"""TRIX trend-following strategy.

TRIX is the rate of change of a triple-smoothed EMA — a low-noise momentum
oscillator that filters out the short-term wiggles a single EMA still passes
through. This strategy goes long while TRIX is positive (the smoothed trend is
rising) and short while it is negative, with an optional signal-line (EMA of
TRIX) crossover mode for earlier, more reactive entries.

Reuses the shared ``trix`` / ``ema`` primitives from :mod:`src.indicators`.
"""

from __future__ import annotations

import logging

import pandas as pd

from src.indicators import ema, trix

logger = logging.getLogger(__name__)


def trix_strategy(
    df: pd.DataFrame,
    period: int = 15,
    signal_period: int = 9,
    use_signal_line: bool = False,
    allow_short: bool = True,
) -> pd.DataFrame:
    """TRIX trend-following signals.

    Signal logic (zero-line mode, the default):
        +1 (long)  when TRIX > 0
        -1 (short) when TRIX < 0
         0 (flat)  otherwise (including the warm-up window)

    With ``use_signal_line=True`` the comparison threshold becomes an EMA of
    TRIX instead of zero, giving earlier crossover entries at the cost of more
    whipsaw.

    Args:
        df: DataFrame with a 'close' column.
        period: EMA length used for the triple smoothing.
        signal_period: EMA span of the TRIX signal line (only used when
            ``use_signal_line`` is True).
        use_signal_line: Compare TRIX against its signal line instead of 0.
        allow_short: If False, short signals are clamped to flat.

    Returns:
        DataFrame with a ``trix`` column (plus ``trix_signal`` in signal-line
        mode) and a ``signal`` column.

    Raises:
        ValueError: If 'close' is missing or a period is < 1.
    """
    if "close" not in df.columns:
        raise ValueError("DataFrame must contain a 'close' column.")
    if period < 1:
        raise ValueError(f"period must be >= 1, got {period}.")
    if use_signal_line and signal_period < 1:
        raise ValueError(f"signal_period must be >= 1, got {signal_period}.")

    df = df.copy()
    trix_line = trix(df["close"], period)
    df["trix"] = trix_line

    if use_signal_line:
        threshold = ema(trix_line, signal_period)
        df["trix_signal"] = threshold
    else:
        threshold = pd.Series(0.0, index=df.index)

    long_cond = trix_line > threshold
    short_cond = trix_line < threshold

    df["signal"] = 0
    df.loc[long_cond, "signal"] = 1
    if allow_short:
        df.loc[short_cond, "signal"] = -1

    return df
