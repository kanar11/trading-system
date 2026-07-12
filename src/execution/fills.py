"""Vectorised limit-order fill simulation on OHLC bars.

Backtesting a limit-order tactic (buy the dip at yesterday's low, sell
into a target) needs an answer per bar: *would this limit have filled,
and at what price?* The event engine answers it order by order; this
module answers it vectorised over a whole history, using the same
conservative bar-versus-limit conventions:

* a BUY limit fills when the bar's low trades at or through the price;
  a SELL limit when the high does;
* if the bar *opens* through the limit the fill happens at the open
  (price improvement — you can never fill worse than your limit, and
  gaps fill better, exactly like the engine's gap protection).

Intrabar path is unknowable from OHLC, so touching the extreme is
treated as a fill — the standard optimistic-touch convention; treat
results as an upper bound on fill rates. NaN limit prices mean "no
order that bar".
"""

from __future__ import annotations

import numpy as np
import pandas as pd

_REQUIRED = ("open", "high", "low")


def simulate_limit_fills(
    df: pd.DataFrame,
    limit_price: float | pd.Series,
    side: str = "buy",
) -> pd.DataFrame:
    """Simulate resting limit-order fills bar by bar.

    Args:
        df: DataFrame with positive, NaN-free ``open``/``high``/``low``
            columns.
        limit_price: Limit level — a scalar for a constant level or a
            per-bar Series on the same index (NaN = no order that bar).
        side: ``"buy"`` or ``"sell"``.

    Returns:
        DataFrame aligned to ``df`` with columns:

        * ``filled`` — True when the limit would have executed.
        * ``fill_price`` — execution price (open when the bar opens
          through the limit, else the limit itself); NaN when unfilled.

    Raises:
        ValueError: If required columns are missing, prices are invalid,
            ``side`` is unknown, or a limit Series is misaligned.
    """
    missing = [c for c in _REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"DataFrame must contain columns {missing}.")
    if side not in ("buy", "sell"):
        raise ValueError(f"side must be 'buy' or 'sell', got {side!r}.")

    opens = df["open"].to_numpy(dtype=float)
    highs = df["high"].to_numpy(dtype=float)
    lows = df["low"].to_numpy(dtype=float)
    bars = np.column_stack([opens, highs, lows])
    if np.isnan(bars).any() or (bars <= 0).any():
        raise ValueError("open/high/low prices must be positive and NaN-free.")

    if isinstance(limit_price, pd.Series):
        if not df.index.equals(limit_price.index):
            raise ValueError("limit_price series must share the DataFrame index.")
        limits = limit_price.to_numpy(dtype=float)
    else:
        limits = np.full(len(df), float(limit_price))
    active = ~np.isnan(limits)
    if (limits[active] <= 0).any():
        raise ValueError("limit prices must be > 0.")

    if side == "buy":
        filled = active & (lows <= limits)
        improved = filled & (opens <= limits)
    else:
        filled = active & (highs >= limits)
        improved = filled & (opens >= limits)

    fill_price = np.where(filled, np.where(improved, opens, limits), np.nan)
    return pd.DataFrame({"filled": filled, "fill_price": fill_price}, index=df.index)
