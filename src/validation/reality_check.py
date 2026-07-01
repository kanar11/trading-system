"""White's Reality Check for data-snooping bias.

When the best of N candidate strategies is selected, its apparent edge is
inflated by the selection itself. White's Reality Check (2000) tests the null
that the *best* strategy does not outperform the benchmark once you account for
having searched over all N. A stationary/moving-block bootstrap of the
per-period performance series builds the null distribution of the max statistic.

A small p-value means the best strategy's performance survives the multiple-
testing correction. Complements the PBO estimate and the Deflated Sharpe ratio.
Pure numpy.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class RealityCheckResult:
    """Outcome of White's Reality Check.

    Attributes:
        p_value: Bootstrap p-value for H0 (best strategy is no better than
            chance). Small = the best strategy's edge is real.
        best_strategy: Column index of the best-performing strategy.
        best_statistic: sqrt(T) * mean performance of the best strategy.
    """

    p_value: float
    best_strategy: int
    best_statistic: float


def _block_bootstrap_indices(rng: np.random.Generator, n: int, block_size: int) -> np.ndarray:
    """Moving-block bootstrap row indices of length ``n``."""
    n_blocks = int(np.ceil(n / block_size))
    starts = rng.integers(0, n - block_size + 1, size=n_blocks)
    idx: np.ndarray = np.concatenate([np.arange(s, s + block_size) for s in starts])[:n]
    return idx


def whites_reality_check(
    performance: np.ndarray,
    n_bootstrap: int = 1000,
    block_size: int = 1,
    seed: int | None = None,
) -> RealityCheckResult:
    """Run White's Reality Check over a panel of strategy performance.

    Args:
        performance: (T, N) array of per-period performance for N candidate
            strategies (e.g. excess returns vs a benchmark).
        n_bootstrap: Number of bootstrap resamples.
        block_size: Moving-block length (1 = iid bootstrap; > 1 preserves
            short-range autocorrelation).
        seed: RNG seed for reproducibility.

    Returns:
        A :class:`RealityCheckResult`.

    Raises:
        ValueError: If the panel is not 2-D, empty, or the bootstrap
            parameters are out of range.
    """
    panel = np.asarray(performance, dtype=float)
    if panel.ndim != 2:
        raise ValueError("performance must be 2-D (T periods x N strategies).")
    n_obs, n_strategies = panel.shape
    if n_strategies < 1 or n_obs < 1:
        raise ValueError("performance must have at least one period and strategy.")
    if n_bootstrap < 1:
        raise ValueError(f"n_bootstrap must be >= 1, got {n_bootstrap}.")
    if not 1 <= block_size <= n_obs:
        raise ValueError(f"block_size must be in [1, {n_obs}], got {block_size}.")

    rng = np.random.default_rng(seed)
    means = panel.mean(axis=0)
    root_t = float(np.sqrt(n_obs))
    observed = float((root_t * means).max())
    best = int(np.argmax(means))

    exceed = 0
    for _ in range(n_bootstrap):
        idx = _block_bootstrap_indices(rng, n_obs, block_size)
        boot_means = panel[idx].mean(axis=0)
        v_star = float((root_t * (boot_means - means)).max())
        if v_star >= observed:
            exceed += 1

    return RealityCheckResult(
        p_value=exceed / n_bootstrap,
        best_strategy=best,
        best_statistic=observed,
    )
