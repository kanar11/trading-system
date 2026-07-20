"""Subperiod performance-stability analysis.

Walk-forward tests out-of-sample *refits*; this asks a simpler, orthogonal
question about a single return series: **is the edge steady, or did one
lucky stretch carry the whole record?** The series is cut into contiguous
equal subperiods (calendar-year-like chunks) and the headline statistics
are computed within each. A strategy whose Sharpe is roughly constant
across subperiods is far more trustworthy than one with the same overall
Sharpe concentrated in a single window.

:func:`subperiod_stats` returns the per-window table; :func:`stability_score`
condenses it to two robustness numbers — the fraction of subperiods that
were profitable and the consistency ratio (mean / std of the per-window
Sharpe). Descriptive, in-sample and cheap; a natural companion to the
resampling tests in this package.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _window_stats(returns: pd.Series, periods_per_year: int) -> dict[str, float]:
    """Absolute performance statistics of one return window."""
    mean = float(returns.mean())
    std = float(returns.std(ddof=1)) if len(returns) > 1 else float("nan")
    ann_return = mean * periods_per_year
    ann_vol = std * float(np.sqrt(periods_per_year))
    equity = (1.0 + returns).cumprod()
    drawdown = equity / equity.cummax() - 1.0
    return {
        "n_obs": float(len(returns)),
        "total_return": float(equity.iloc[-1] - 1.0),
        "ann_return": ann_return,
        "ann_vol": ann_vol,
        "sharpe": ann_return / ann_vol if np.isfinite(ann_vol) and ann_vol > 0 else float("nan"),
        "max_drawdown": float(drawdown.min()),
        "hit_rate": float((returns > 0).mean()),
    }


def subperiod_stats(
    returns: pd.Series,
    n_periods: int = 4,
    periods_per_year: int = 252,
) -> pd.DataFrame:
    """Per-subperiod performance table over contiguous windows.

    Args:
        returns: Per-bar return series (any index; order is what matters).
        n_periods: Number of contiguous equal subperiods (>= 2, <= len).
        periods_per_year: Bars per year for annualisation.

    Returns:
        DataFrame indexed ``period_0 .. period_{n-1}`` with columns
        ``n_obs``, ``total_return``, ``ann_return``, ``ann_vol``,
        ``sharpe``, ``max_drawdown`` and ``hit_rate``.

    Raises:
        ValueError: If ``returns`` is empty, ``n_periods`` is out of range,
            or ``periods_per_year`` < 1.
    """
    if len(returns) == 0:
        raise ValueError("returns must not be empty.")
    if not 2 <= n_periods <= len(returns):
        raise ValueError(f"n_periods must be in [2, {len(returns)}], got {n_periods}.")
    if periods_per_year < 1:
        raise ValueError(f"periods_per_year must be >= 1, got {periods_per_year}.")

    chunks = np.array_split(np.arange(len(returns)), n_periods)
    rows = {
        f"period_{i}": _window_stats(returns.iloc[chunk], periods_per_year)
        for i, chunk in enumerate(chunks)
    }
    table = pd.DataFrame.from_dict(rows, orient="index")
    table.index.name = "period"
    return table


def stability_score(
    returns: pd.Series,
    n_periods: int = 4,
    periods_per_year: int = 252,
) -> dict[str, float]:
    """Condense the subperiod table into robustness numbers.

    Args:
        returns: Per-bar return series.
        n_periods: Number of contiguous subperiods (see
            :func:`subperiod_stats`).
        periods_per_year: Bars per year for annualisation.

    Returns:
        Dict with:

        * ``positive_fraction`` — share of subperiods with a positive
          total return (1.0 = profitable in every window).
        * ``sharpe_consistency`` — ``mean(sharpe) / std(sharpe)`` across
          subperiods (higher = steadier; NaN if fewer than two finite
          Sharpes or zero dispersion).
        * ``worst_sharpe`` / ``best_sharpe`` — the range of per-window
          Sharpes.

    Raises:
        ValueError: As for :func:`subperiod_stats`.
    """
    table = subperiod_stats(returns, n_periods=n_periods, periods_per_year=periods_per_year)
    sharpes = table["sharpe"].to_numpy()
    finite = sharpes[np.isfinite(sharpes)]

    if len(finite) >= 2:
        std = float(finite.std(ddof=1))
        consistency = float(finite.mean() / std) if std > 0 else float("nan")
    else:
        consistency = float("nan")

    return {
        "positive_fraction": float((table["total_return"] > 0).mean()),
        "sharpe_consistency": consistency,
        "worst_sharpe": float(np.nanmin(sharpes)) if len(finite) else float("nan"),
        "best_sharpe": float(np.nanmax(sharpes)) if len(finite) else float("nan"),
    }
