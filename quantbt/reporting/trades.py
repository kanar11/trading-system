"""Trade log builder — standalone utility.

This module is kept for ad-hoc analysis of custom DataFrames that use
a 'pos' column convention. The primary pipeline uses the trade log
built inside ``src.backtest.engine``.

For normal backtests you do NOT need to call this module directly.
"""

import logging
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


def build_trade_log(result: pd.DataFrame) -> pd.DataFrame:
    """Build a trade-level log from a DataFrame with a 'pos' column.

    This is a standalone utility for DataFrames that follow the 'pos'
    column convention (integer direction: +1 long, -1 short, 0 flat).
    The main backtest pipeline builds its own trade log internally.

    Args:
        result: DataFrame with 'close' and 'pos' columns, indexed by date.

    Returns:
        DataFrame with one row per round-trip trade containing entry/exit
        dates, prices, direction, return, and holding period. Empty if the
        input has no position changes.

    Raises:
        ValueError: If the 'pos' or 'close' column is missing.
    """
    df = result.copy()

    for col in ("pos", "close"):
        if col not in df.columns:
            raise ValueError(
                f"DataFrame must contain a '{col}' column. For backtest "
                "results use the trade log from backtest_strategy()."
            )

    pos = df["pos"].astype(int)
    prev_pos = pos.shift(1).fillna(0).astype(int)

    change = pos - prev_pos
    change_dates = df.index[change != 0]

    trades: list[dict[str, Any]] = []
    current_pos: int = 0
    entry_date: pd.Timestamp | None = None
    entry_price: float | None = None

    for dt in change_dates:
        new_pos = int(pos.loc[dt])
        price = float(df.loc[dt, "close"])

        # close existing trade
        if current_pos != 0 and entry_price is not None:
            trades.append(_make_trade(current_pos, entry_date, dt, entry_price, price))
            entry_date = None
            entry_price = None

        # open new trade
        if new_pos != 0:
            current_pos = new_pos
            entry_date = dt
            entry_price = price
        else:
            current_pos = 0

    # mark-to-market open position at the end
    if current_pos != 0 and entry_price is not None:
        last_date = df.index[-1]
        last_price = float(df["close"].iloc[-1])
        trades.append(_make_trade(current_pos, entry_date, last_date, entry_price, last_price))

    trade_df = pd.DataFrame(trades)

    if not trade_df.empty:
        trade_df["holding_days"] = (trade_df["exit_date"] - trade_df["entry_date"]).dt.days

    return trade_df


def _make_trade(
    direction: int,
    entry_date: pd.Timestamp | None,
    exit_date: pd.Timestamp,
    entry_price: float,
    exit_price: float,
) -> dict[str, Any]:
    """Assemble a single round-trip trade record."""
    return {
        "entry_date": entry_date,
        "exit_date": exit_date,
        "direction": direction,
        "entry_price": entry_price,
        "exit_price": exit_price,
        "trade_return": direction * (exit_price / entry_price - 1.0),
    }
