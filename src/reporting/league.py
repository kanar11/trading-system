"""Multi-strategy league table.

:func:`src.reporting.benchmark.benchmark_comparison` grades one strategy
against one benchmark; a research session ends with a *dozen* candidates
(parameter variants, the sweep output, every template in
:mod:`src.strategy`) that need one sortable table. This builds it: one
row per strategy with the absolute statistics block, plus beta and
information ratio versus a common benchmark when one is supplied —
ranked by the metric of your choice.

Direct-import module::

    from src.reporting.league import strategy_league
"""

from __future__ import annotations

from collections.abc import Mapping

import pandas as pd

from src.reporting.attribution import compute_beta
from src.reporting.benchmark import _absolute_stats
from src.risk.metrics import information_ratio, tracking_error


def strategy_league(
    returns: Mapping[str, pd.Series],
    benchmark: pd.Series | None = None,
    periods_per_year: int = 252,
    sort_by: str = "sharpe",
) -> pd.DataFrame:
    """Rank a set of strategies on the standard statistics block.

    Args:
        returns: ``{strategy name: per-bar return series}`` (non-empty;
            each series non-empty). Series may cover different periods —
            each row is computed on its own history.
        benchmark: Optional common benchmark; adds ``beta`` and
            ``information_ratio`` columns, computed on each strategy's
            overlap with it (inner alignment of the underlying helpers).
        periods_per_year: Bars per year for annualisation.
        sort_by: Column to rank by, descending (NaNs last).

    Returns:
        DataFrame indexed by strategy name, sorted by ``sort_by``, with
        ``ann_return``, ``ann_vol``, ``sharpe``, ``max_drawdown``,
        ``hit_rate``, ``n_obs`` (+ ``beta``/``information_ratio`` when a
        benchmark is given).

    Raises:
        ValueError: If the mapping or any series is empty,
            ``periods_per_year`` < 1, or ``sort_by`` is not a column.
    """
    if not returns:
        raise ValueError("returns must not be empty.")
    if periods_per_year < 1:
        raise ValueError(f"periods_per_year must be >= 1, got {periods_per_year}.")

    rows: dict[str, dict[str, float]] = {}
    for name, series in returns.items():
        if len(series) == 0:
            raise ValueError(f"return series for {name!r} is empty.")
        stats = _absolute_stats(series, periods_per_year)
        stats["n_obs"] = float(len(series))
        if benchmark is not None:
            stats["beta"] = compute_beta(series, benchmark)
            te = tracking_error(series, benchmark, periods_per_year=periods_per_year)
            stats["information_ratio"] = (
                information_ratio(series, benchmark, periods_per_year) if te > 0 else float("nan")
            )
        rows[name] = stats

    table = pd.DataFrame.from_dict(rows, orient="index")
    table.index.name = "strategy"
    if sort_by not in table.columns:
        raise ValueError(f"sort_by must be one of {list(table.columns)}, got {sort_by!r}.")
    return table.sort_values(sort_by, ascending=False, na_position="last")
