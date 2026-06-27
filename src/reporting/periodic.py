"""Periodic-return analytics: calendar tables and rolling metrics.

Standard tear-sheet companions that summarise a daily return stream over
calendar periods — a year x month table with an annual total, per-year
returns, and rolling annualised performance metrics. All expect a Series
indexed by a DatetimeIndex. Pure pandas; inputs are never mutated.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def annual_returns(returns: pd.Series) -> pd.Series:
    """Compounded return for each calendar year.

    Args:
        returns: Daily returns indexed by date.

    Returns:
        Series of yearly returns indexed by year (named ``annual_return``).
    """
    r = pd.Series(returns).dropna()
    if r.empty:
        return pd.Series(dtype=float, name="annual_return")
    years = pd.DatetimeIndex(r.index).year
    out: pd.Series = (1 + r).groupby(years).prod() - 1
    out.index.name = "year"
    return out.rename("annual_return")


def monthly_returns_table(returns: pd.Series) -> pd.DataFrame:
    """Year x month table of compounded returns plus an annual total column.

    Args:
        returns: Daily returns indexed by date.

    Returns:
        DataFrame indexed by year with integer month columns (1-12) and a
        trailing ``annual`` column (the compounded full-year return). Empty
        DataFrame when there are no returns.
    """
    r = pd.Series(returns).dropna()
    if r.empty:
        return pd.DataFrame()
    monthly = (1 + r).resample("ME").prod() - 1
    idx = pd.DatetimeIndex(monthly.index)
    frame = pd.DataFrame({"ret": monthly.to_numpy(), "year": idx.year, "month": idx.month})
    table = frame.pivot(index="year", columns="month", values="ret")
    month_cols = list(table.columns)
    table["annual"] = (1 + table[month_cols]).prod(axis=1, min_count=1) - 1
    return table


def rolling_metrics(
    returns: pd.Series,
    window: int = 63,
    periods_per_year: int = 252,
) -> pd.DataFrame:
    """Rolling annualised return, volatility, and Sharpe.

    Args:
        returns: Daily returns indexed by date.
        window: Rolling window length in periods (>= 2).
        periods_per_year: Annualisation factor (252 for daily data).

    Returns:
        DataFrame with ``return``, ``volatility`` and ``sharpe`` columns; the
        first ``window - 1`` rows are NaN.

    Raises:
        ValueError: If ``window`` < 2.
    """
    if window < 2:
        raise ValueError(f"window must be >= 2, got {window}.")
    r = pd.Series(returns).dropna()
    mean = r.rolling(window).mean()
    std = r.rolling(window).std(ddof=1)
    ann = float(periods_per_year)
    sharpe = (mean / std.replace(0, np.nan)) * (ann**0.5)
    return pd.DataFrame(
        {
            "return": mean * ann,
            "volatility": std * (ann**0.5),
            "sharpe": sharpe,
        }
    )
