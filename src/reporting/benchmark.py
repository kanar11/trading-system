"""Strategy-vs-benchmark comparison table.

Every momentum paper and fund factsheet leads with the same head-to-head
table: the strategy and its benchmark (SPY, an index) side by side on
absolute statistics, plus the benchmark-relative block — beta, Jensen's
alpha, tracking error, information ratio and capture ratios. This module
assembles that table from the primitives already in the package
(:mod:`src.risk.metrics`, :mod:`src.reporting.attribution`) so a TSMOM
overlay can be judged against buy-and-hold in one call.

Absolute rows are computed for both columns; relative rows only make
sense for the strategy and are NaN in the benchmark column. Direct-import
module::

    from src.reporting.benchmark import benchmark_comparison
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.reporting.attribution import compute_beta, down_capture, up_capture
from src.risk.metrics import information_ratio, jensen_alpha, tracking_error

_RELATIVE_ROWS = (
    "beta",
    "jensen_alpha",
    "tracking_error",
    "information_ratio",
    "up_capture",
    "down_capture",
    "correlation",
)


def _annualised_return(returns: pd.Series, periods_per_year: int) -> float:
    growth = float((1.0 + returns).to_numpy().prod())
    if growth <= 0:
        return float("nan")
    return float(growth ** (periods_per_year / len(returns))) - 1.0


def _absolute_stats(returns: pd.Series, periods_per_year: int) -> dict[str, float]:
    ann_vol = float(returns.std(ddof=1)) * float(np.sqrt(periods_per_year))
    mean_ann = float(returns.mean()) * periods_per_year
    equity = (1.0 + returns).cumprod()
    drawdown = equity / equity.cummax() - 1.0
    return {
        "ann_return": _annualised_return(returns, periods_per_year),
        "ann_vol": ann_vol,
        "sharpe": mean_ann / ann_vol if ann_vol > 0 else float("nan"),
        "max_drawdown": float(drawdown.min()),
        "hit_rate": float((returns > 0).mean()),
    }


def benchmark_comparison(
    strategy_returns: pd.Series,
    benchmark_returns: pd.Series,
    periods_per_year: int = 252,
) -> pd.DataFrame:
    """Head-to-head statistics table for a strategy and its benchmark.

    Args:
        strategy_returns: Per-bar strategy returns.
        benchmark_returns: Per-bar benchmark returns on the same index.
        periods_per_year: Bars per year for annualisation (the CAPM rows
            from :mod:`src.risk.metrics` assume daily bars).

    Returns:
        DataFrame with columns ``strategy`` / ``benchmark``. Absolute rows
        (``ann_return``, ``ann_vol``, ``sharpe``, ``max_drawdown``,
        ``hit_rate``) are filled for both; the benchmark-relative block
        (``beta``, ``jensen_alpha``, ``tracking_error``,
        ``information_ratio``, ``up_capture``, ``down_capture``,
        ``correlation``) only for the strategy.

    Raises:
        ValueError: If the indexes differ, the series are empty, or
            ``periods_per_year`` < 1.
    """
    if len(strategy_returns) == 0:
        raise ValueError("strategy_returns must not be empty.")
    if not strategy_returns.index.equals(benchmark_returns.index):
        raise ValueError("strategy and benchmark returns must share the same index.")
    if periods_per_year < 1:
        raise ValueError(f"periods_per_year must be >= 1, got {periods_per_year}.")

    table = pd.DataFrame(
        {
            "strategy": _absolute_stats(strategy_returns, periods_per_year),
            "benchmark": _absolute_stats(benchmark_returns, periods_per_year),
        }
    )

    te = tracking_error(strategy_returns, benchmark_returns, periods_per_year=periods_per_year)
    relative = {
        "beta": compute_beta(strategy_returns, benchmark_returns),
        "jensen_alpha": jensen_alpha(strategy_returns, benchmark_returns),
        "tracking_error": te,
        "information_ratio": (
            information_ratio(strategy_returns, benchmark_returns, periods_per_year)
            if te > 0
            else float("nan")
        ),
        "up_capture": up_capture(strategy_returns, benchmark_returns),
        "down_capture": down_capture(strategy_returns, benchmark_returns),
        "correlation": float(strategy_returns.corr(benchmark_returns)),
    }
    for row in _RELATIVE_ROWS:
        table.loc[row, "strategy"] = relative[row]
        table.loc[row, "benchmark"] = float("nan")
    return table
