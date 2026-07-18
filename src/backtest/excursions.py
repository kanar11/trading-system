"""Per-trade MAE / MFE excursion analysis.

A trade's entry-to-exit return hides its whole life: how far it went
against you first (Maximum Adverse Excursion) and how much open profit
it showed at its best (Maximum Favorable Excursion). Sweeney's classic
MAE/MFE analysis reads stop and target placement straight off those
distributions — stops just beyond the MAE cluster of winners, targets
near typical MFE — and the ``efficiency`` ratio (realised return over
MFE) shows how much of the available move exits actually captured.

Consumes the trade log of :func:`src.backtest.engine.backtest_strategy`
plus the OHLC frame it ran on; excursions are measured on intrabar
highs/lows over each trade's holding window, side-aware.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

_TRADE_COLUMNS = ("entry_date", "exit_date", "direction", "entry_price")


def trade_excursions(df: pd.DataFrame, trade_log: pd.DataFrame) -> pd.DataFrame:
    """Annotate a trade log with MAE, MFE and capture efficiency.

    Args:
        df: OHLC frame the backtest ran on (needs ``high``/``low``).
        trade_log: The engine's trade log (``entry_date``, ``exit_date``,
            ``direction`` ±1, ``entry_price``; extra columns pass
            through). May be empty.

    Returns:
        Copy of ``trade_log`` with three columns added, all as returns on
        the entry price:

        * ``mae`` — worst open drawdown during the trade (<= 0).
        * ``mfe`` — best open profit during the trade (>= 0).
        * ``efficiency`` — ``trade_return / mfe`` when both available and
          MFE > 0, else NaN (how much of the best move the exit kept).

    Raises:
        ValueError: If required columns are missing or a trade's window
            has no bars in ``df``.
    """
    missing = [c for c in ("high", "low") if c not in df.columns]
    if missing:
        raise ValueError(f"DataFrame must contain columns {missing}.")
    missing_log = [c for c in _TRADE_COLUMNS if c not in trade_log.columns]
    if missing_log:
        raise ValueError(f"trade_log must contain columns {missing_log}.")

    out = trade_log.copy()
    if out.empty:
        out["mae"] = pd.Series(dtype=float)
        out["mfe"] = pd.Series(dtype=float)
        out["efficiency"] = pd.Series(dtype=float)
        return out

    entries = out["entry_date"].to_numpy()
    exits = out["exit_date"].to_numpy()
    directions = out["direction"].to_numpy(dtype=int)
    entry_prices = out["entry_price"].to_numpy(dtype=float)

    mae = np.empty(len(out))
    mfe = np.empty(len(out))
    for i in range(len(out)):
        window = df.loc[entries[i] : exits[i]]
        if window.empty:
            raise ValueError(f"no bars between {entries[i]} and {exits[i]} in the frame.")
        entry = entry_prices[i]
        highest = float(window["high"].max())
        lowest = float(window["low"].min())
        if directions[i] == 1:
            mfe[i] = highest / entry - 1.0
            mae[i] = lowest / entry - 1.0
        else:
            mfe[i] = entry / lowest - 1.0
            mae[i] = entry / highest - 1.0

    out["mae"] = np.minimum(mae, 0.0)  # a trade that never went adverse: 0
    out["mfe"] = np.maximum(mfe, 0.0)
    if "trade_return" in out.columns:
        with np.errstate(divide="ignore", invalid="ignore"):
            out["efficiency"] = np.where(
                out["mfe"].to_numpy() > 0,
                out["trade_return"].to_numpy(dtype=float) / out["mfe"].to_numpy(),
                np.nan,
            )
    else:
        out["efficiency"] = np.nan
    return out
