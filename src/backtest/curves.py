"""Equity and drawdown curves from a returns series.

Foundational transforms that turn a per-bar return series into a cumulative
equity curve and its underwater (drawdown-from-peak) curve. Reusable across the
vectorised engine output, combined walk-forward OOS returns, or portfolio
returns. Pure pandas; the input is not mutated.
"""

from __future__ import annotations

import pandas as pd


def equity_curve(returns: pd.Series, initial: float = 1.0) -> pd.Series:
    """Cumulative equity from a return series: ``initial * prod(1 + r)``.

    Missing returns are treated as 0 (no move). An empty input returns an empty
    Series.
    """
    r = pd.Series(returns).fillna(0.0)
    out: pd.Series = initial * (1.0 + r).cumprod()
    return out


def drawdown_series(returns: pd.Series) -> pd.Series:
    """Underwater curve: equity divided by its running peak, minus 1 (<= 0).

    0 at each new high; increasingly negative through a drawdown. The minimum is
    the maximum drawdown.
    """
    eq = equity_curve(returns)
    if eq.empty:
        return eq
    out: pd.Series = eq / eq.cummax() - 1.0
    return out
