"""Monte-Carlo robustness analysis.

Walk-forward measures degradation across time; Monte Carlo measures
degradation under reshuffling. Both are useful, and they answer
different questions:

    * Walk-forward: "did the strategy survive different market regimes?"
    * Monte Carlo:  "how much of the equity curve was luck of ordering?"

Two routines are provided:

    * ``bootstrap_returns``: resample the daily return series with
      replacement to produce a confidence interval over performance
      metrics (Sharpe, max drawdown, total return).
    * ``shuffle_trade_log``: permute the order of completed trades to
      check whether the equity curve is driven by a small number of
      lucky sequences.

Both use a fixed numpy RNG for reproducibility.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.reporting.metrics import calculate_metrics

logger = logging.getLogger(__name__)


@dataclass
class MonteCarloResult:
    """Aggregate output of a Monte-Carlo run.

    Attributes:
        n_simulations: Number of bootstrap iterations performed.
        metric_samples: DataFrame, one row per simulation, columns are
            the metrics from :func:`calculate_metrics`.
        summary: Mean / std / percentile table for each metric.
    """

    n_simulations: int
    metric_samples: pd.DataFrame
    summary: pd.DataFrame


def bootstrap_returns(
    returns: pd.Series,
    n_simulations: int = 1000,
    block_size: int = 1,
    seed: int | None = 42,
) -> MonteCarloResult:
    """Bootstrap-resample a return series and recompute metrics.

    With ``block_size=1`` this is a plain i.i.d. bootstrap, suitable
    for trade-return series where serial correlation is small. For
    daily-bar series set ``block_size`` to a small value (e.g. 5-20)
    to use a moving-block bootstrap that preserves short-range
    autocorrelation.

    Args:
        returns: Original return series (daily or per-trade).
        n_simulations: Number of bootstrap samples to draw.
        block_size: Block length for the moving-block bootstrap.
            Use 1 for i.i.d. resampling.
        seed: RNG seed for reproducibility (set None for random).

    Returns:
        A :class:`MonteCarloResult` with per-simulation metrics and a
        summary table (mean / std / 5%/50%/95%).
    """
    r = pd.Series(returns).dropna().reset_index(drop=True)
    n = len(r)
    if n == 0:
        raise ValueError("returns series is empty")
    if block_size < 1:
        raise ValueError("block_size must be >= 1")
    if n_simulations < 1:
        raise ValueError("n_simulations must be >= 1")

    rng = np.random.default_rng(seed)
    values = r.values

    metric_rows: list[dict[str, float]] = []
    for _ in range(n_simulations):
        if block_size == 1:
            idx = rng.integers(0, n, size=n)
            sample = values[idx]
        else:
            n_blocks = int(np.ceil(n / block_size))
            starts = rng.integers(0, max(n - block_size + 1, 1), size=n_blocks)
            blocks = [values[s:s + block_size] for s in starts]
            sample = np.concatenate(blocks)[:n]

        metric_rows.append(calculate_metrics(pd.Series(sample)))

    metric_df = pd.DataFrame(metric_rows)
    summary = pd.DataFrame(
        {
            "mean": metric_df.mean(),
            "std": metric_df.std(ddof=1),
            "p05": metric_df.quantile(0.05),
            "p50": metric_df.quantile(0.50),
            "p95": metric_df.quantile(0.95),
        }
    )

    return MonteCarloResult(
        n_simulations=n_simulations,
        metric_samples=metric_df,
        summary=summary,
    )


def shuffle_trade_log(
    trade_returns: pd.Series,
    n_simulations: int = 1000,
    seed: int | None = 42,
) -> MonteCarloResult:
    """Shuffle the order of completed trades and recompute metrics.

    Unlike :func:`bootstrap_returns` this samples WITHOUT replacement
    — it keeps the exact same set of trades and only randomises the
    sequence. Useful for separating "edge" (set of trades) from
    "luck" (ordering effects on path-dependent metrics like max
    drawdown).

    Args:
        trade_returns: Per-trade return series (e.g. trade_log['trade_return']).
        n_simulations: Number of permutations to draw.
        seed: RNG seed for reproducibility.

    Returns:
        A :class:`MonteCarloResult` with per-simulation metrics.
    """
    r = pd.Series(trade_returns).dropna().reset_index(drop=True)
    if len(r) == 0:
        raise ValueError("trade_returns series is empty")

    rng = np.random.default_rng(seed)
    values = r.values

    rows: list[dict[str, float]] = []
    for _ in range(n_simulations):
        perm = rng.permutation(values)
        rows.append(calculate_metrics(pd.Series(perm)))

    metric_df = pd.DataFrame(rows)
    summary = pd.DataFrame(
        {
            "mean": metric_df.mean(),
            "std": metric_df.std(ddof=1),
            "p05": metric_df.quantile(0.05),
            "p50": metric_df.quantile(0.50),
            "p95": metric_df.quantile(0.95),
        }
    )

    return MonteCarloResult(
        n_simulations=n_simulations,
        metric_samples=metric_df,
        summary=summary,
    )


def print_monte_carlo_report(result: MonteCarloResult, title: str = "Monte Carlo") -> None:
    """Pretty-print a Monte-Carlo summary table."""
    print("\n" + "=" * 60)
    print(f"{title} — {result.n_simulations} simulations")
    print("=" * 60)
    formatted = result.summary.copy()
    for col in formatted.columns:
        formatted[col] = formatted[col].map(lambda x: f"{x:.4f}")
    print(formatted.to_string())
    print("=" * 60)
