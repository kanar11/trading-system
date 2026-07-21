"""Bootstrap equity confidence cone (fan chart).

A single backtest equity curve is one draw from a distribution of paths
the same edge could have produced. The confidence cone makes that
distribution visible: resample the return series many times with a
moving-block bootstrap (preserving short-range autocorrelation), compound
each resample into an equity path, and read off the percentile band at
every bar. The result is the familiar widening "cone of uncertainty" — a
median trajectory flanked by e.g. 5th/95th-percentile edges that fan out
with the horizon.

It answers the question a point estimate cannot: *how good or bad could
this have plausibly looked?* A strategy whose 5th-percentile path still
ends above water is far more robust than one with the same median but a
cone that dips deep into losses.

Direct-import module::

    from src.reporting.mc_cone import equity_cone
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _block_bootstrap_sample(
    rng: np.random.Generator, values: np.ndarray, block_size: int
) -> np.ndarray:
    """One moving-block bootstrap resample of ``values`` (same length)."""
    n = len(values)
    n_blocks = int(np.ceil(n / block_size))
    starts = rng.integers(0, max(n - block_size + 1, 1), size=n_blocks)
    idx = np.concatenate([np.arange(s, s + block_size) for s in starts])[:n]
    sample: np.ndarray = values[idx]
    return sample


def equity_cone(
    returns: pd.Series,
    n_simulations: int = 1000,
    block_size: int = 1,
    percentiles: tuple[float, ...] = (5.0, 25.0, 50.0, 75.0, 95.0),
    initial: float = 1.0,
    seed: int | None = 42,
) -> pd.DataFrame:
    """Percentile bands of bootstrapped equity paths over the horizon.

    Each simulation moving-block-resamples ``returns`` and compounds it
    into an equity path starting at ``initial``; the returned table is the
    per-bar percentile across simulations, so column ``p50`` is the median
    trajectory and the outer columns are the cone edges.

    Args:
        returns: Per-bar return series (NaN-free).
        n_simulations: Number of bootstrap paths (>= 1).
        block_size: Moving-block length (1 = iid; > 1 preserves short-range
            autocorrelation).
        percentiles: Percentile levels to report, each in (0, 100).
        initial: Starting equity level (> 0).
        seed: RNG seed for reproducibility.

    Returns:
        DataFrame indexed like ``returns`` with one ``p{level}`` column per
        percentile; every row's values are non-decreasing across the
        (sorted) percentile columns. Row 0 equals ``initial`` in every
        column (equity before the first return).

    Raises:
        ValueError: If ``returns`` is empty or has NaNs, the counts are out
            of range, ``initial`` <= 0, or a percentile is outside (0, 100).
    """
    values = returns.to_numpy(dtype=float)
    if len(values) == 0:
        raise ValueError("returns must not be empty.")
    if np.isnan(values).any():
        raise ValueError("returns must not contain NaNs.")
    if n_simulations < 1:
        raise ValueError(f"n_simulations must be >= 1, got {n_simulations}.")
    if not 1 <= block_size <= len(values):
        raise ValueError(f"block_size must be in [1, {len(values)}], got {block_size}.")
    if initial <= 0:
        raise ValueError(f"initial must be > 0, got {initial}.")
    if not all(0.0 < p < 100.0 for p in percentiles):
        raise ValueError("every percentile must be in (0, 100).")

    rng = np.random.default_rng(seed)
    n = len(values)
    paths = np.empty((n_simulations, n))
    for s in range(n_simulations):
        sample = _block_bootstrap_sample(rng, values, block_size)
        paths[s] = initial * np.cumprod(1.0 + sample)

    levels = sorted(percentiles)
    band = np.percentile(paths, levels, axis=0)  # shape (n_levels, n)
    table = pd.DataFrame(
        band.T,
        index=returns.index,
        columns=[f"p{level:g}" for level in levels],
    )
    # anchor the cone at the starting level (equity before any return)
    table.iloc[0] = initial
    return table
