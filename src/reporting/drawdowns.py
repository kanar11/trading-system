"""Worst-drawdown episode table.

Decomposes an equity path into distinct peak-to-trough-to-recovery drawdown
episodes and tabulates the deepest ones — the standard "worst drawdowns" panel
of a tear-sheet. Complements the single max-drawdown stat in
:mod:`src.risk.metrics` and the underwater curve in :mod:`src.backtest.curves`.

Pure pandas; the input is not mutated.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

_COLUMNS = ["peak_date", "trough_date", "recovery_date", "depth", "length"]


def drawdown_table(returns: pd.Series, top_n: int = 5) -> pd.DataFrame:
    """Tabulate the ``top_n`` deepest drawdown episodes of a return series.

    An episode runs from an equity peak, through its lowest point, to the bar
    that first regains the peak (``recovery_date`` is ``NaT`` if it never does).

    Args:
        returns: Per-bar returns indexed by date.
        top_n: Maximum number of episodes to return (deepest first).

    Returns:
        DataFrame with columns ``peak_date``, ``trough_date``,
        ``recovery_date``, ``depth`` (peak-to-trough return, negative) and
        ``length`` (bars from peak to recovery / series end), sorted by depth.
    """
    r = pd.Series(returns).dropna()
    if r.empty:
        return pd.DataFrame(columns=_COLUMNS)

    equity = (1.0 + r).cumprod()
    index = equity.index
    values = equity.to_numpy()
    n = len(values)

    episodes: list[tuple[int, int, int | None]] = []  # (peak_i, trough_i, recovery_i)
    peak = values[0]
    peak_i = 0
    trough_i = 0
    in_drawdown = False

    for i in range(n):
        v = values[i]
        if v >= peak:
            if in_drawdown:
                episodes.append((peak_i, trough_i, i))
                in_drawdown = False
            peak = v
            peak_i = i
        else:
            if not in_drawdown:
                in_drawdown = True
                trough_i = i
            if v < values[trough_i]:
                trough_i = i
    if in_drawdown:
        episodes.append((peak_i, trough_i, None))

    records: list[dict[str, Any]] = []
    for p_i, t_i, rec_i in episodes:
        end_i = rec_i if rec_i is not None else n - 1
        records.append(
            {
                "peak_date": index[p_i],
                "trough_date": index[t_i],
                "recovery_date": index[rec_i] if rec_i is not None else pd.NaT,
                "depth": float(values[t_i] / values[p_i] - 1.0),
                "length": int(end_i - p_i),
            }
        )

    table = pd.DataFrame(records, columns=_COLUMNS)
    return table.sort_values("depth").head(top_n).reset_index(drop=True)
