"""Combinatorial Purged Cross-Validation (CPCV).

Standard (purged) K-fold tests each observation exactly once, producing a
single backtest path. López de Prado's CPCV (Advances in Financial Machine
Learning, 2018, ch. 12) instead partitions the sample into ``n_groups``
contiguous blocks and forms one split for *every combination* of
``n_test_groups`` test blocks — C(N, k) splits in total, from which
φ = C(N−1, k−1) full backtest paths can be assembled. More paths mean a
distribution of out-of-sample performance instead of a point estimate, which
is the raw material for PBO-style overfit diagnostics
(see :mod:`src.validation.pbo`).

Leakage control matches :mod:`src.validation.purged_cv`: training
observations in the ``purge`` bars immediately before each test block and in
the ``embargo`` span immediately after it are dropped. Pure numpy,
index-based, composes with any model or backtest.
"""

from __future__ import annotations

import math
from itertools import combinations

import numpy as np


def n_backtest_paths(n_groups: int, n_test_groups: int) -> int:
    """Number of full backtest paths CPCV generates, φ = C(N−1, k−1).

    Each group appears in C(N−1, k−1) of the C(N, k) test sets, so the
    per-group forecasts can be woven into exactly that many complete paths.

    Args:
        n_groups: Number of contiguous blocks N (>= 2).
        n_test_groups: Test blocks per split k (1 <= k < N).

    Returns:
        The path count φ.

    Raises:
        ValueError: If the group counts are out of range.
    """
    _validate_groups(n_groups, n_test_groups)
    return math.comb(n_groups - 1, n_test_groups - 1)


def combinatorial_purged_splits(
    n_samples: int,
    n_groups: int = 6,
    n_test_groups: int = 2,
    embargo: float = 0.0,
    purge: int = 0,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Generate all C(N, k) purged & embargoed CPCV train/test splits.

    The sample ``[0, n_samples)`` is cut into ``n_groups`` contiguous blocks;
    each split takes one combination of ``n_test_groups`` blocks as the test
    set. Training indices falling in the ``purge`` bars immediately before
    any test block, or in the embargo span immediately after one, are
    removed.

    Args:
        n_samples: Number of ordered observations.
        n_groups: Number of contiguous blocks N (2 <= N <= n_samples).
        n_test_groups: Test blocks per split k (1 <= k < N).
        embargo: Embargo span after each test block, as a fraction of
            ``n_samples`` (e.g. 0.01 = 1%). Rounded up to whole bars.
        purge: Number of bars before each test block to purge from training.

    Returns:
        A list of ``(train_idx, test_idx)`` numpy arrays, one per
        combination, in :func:`itertools.combinations` order.

    Raises:
        ValueError: If the group counts are out of range or ``embargo`` /
            ``purge`` is negative.
    """
    _validate_groups(n_groups, n_test_groups)
    if n_groups > n_samples:
        raise ValueError(f"n_groups ({n_groups}) cannot exceed n_samples ({n_samples}).")
    if embargo < 0:
        raise ValueError(f"embargo must be >= 0, got {embargo}.")
    if purge < 0:
        raise ValueError(f"purge must be >= 0, got {purge}.")

    indices = np.arange(n_samples)
    blocks = np.array_split(indices, n_groups)
    embargo_size = int(np.ceil(embargo * n_samples))

    splits: list[tuple[np.ndarray, np.ndarray]] = []
    for combo in combinations(range(n_groups), n_test_groups):
        train_mask = np.ones(n_samples, dtype=bool)
        for g in combo:
            start = int(blocks[g][0])
            end = int(blocks[g][-1]) + 1
            train_mask[max(start - purge, 0) : start] = False  # purge before
            train_mask[start:end] = False  # the test block itself
            train_mask[end : min(end + embargo_size, n_samples)] = False  # embargo after
        test_idx = np.concatenate([blocks[g] for g in combo])
        splits.append((indices[train_mask], test_idx))
    return splits


def _validate_groups(n_groups: int, n_test_groups: int) -> None:
    """Shared range checks for the CPCV group parameters."""
    if n_groups < 2:
        raise ValueError(f"n_groups must be >= 2, got {n_groups}.")
    if not 1 <= n_test_groups < n_groups:
        raise ValueError(
            f"n_test_groups must be in [1, n_groups), got {n_test_groups} for {n_groups} groups."
        )
