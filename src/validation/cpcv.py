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
from collections.abc import Sequence
from itertools import combinations

import numpy as np
import pandas as pd


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


def assemble_backtest_paths(n_groups: int = 6, n_test_groups: int = 2) -> np.ndarray:
    """Assign every CPCV test forecast to one of the φ backtest paths.

    Across the C(N, k) splits of :func:`combinatorial_purged_splits`, each
    group is a test group exactly φ = C(N−1, k−1) times. Those appearances
    can be woven into φ complete out-of-sample paths: path ``j`` takes each
    group's forecasts from the ``j``-th split (in combination order) that
    tested it. Every (split, test-group) pair is consumed exactly once, so
    the C(N, k) split results reassemble into φ full equity paths — the
    input distribution for PBO-style overfit diagnostics.

    Args:
        n_groups: Number of contiguous blocks N (>= 2).
        n_test_groups: Test blocks per split k (1 <= k < N).

    Returns:
        Integer array of shape ``(φ, n_groups)``: entry ``[j, g]`` is the
        index (into the split list) of the split whose test set supplies
        group ``g``'s forecasts on path ``j``.

    Raises:
        ValueError: If the group counts are out of range.
    """
    _validate_groups(n_groups, n_test_groups)
    phi = math.comb(n_groups - 1, n_test_groups - 1)
    paths = np.empty((phi, n_groups), dtype=int)
    appearances = [0] * n_groups
    for split_idx, combo in enumerate(combinations(range(n_groups), n_test_groups)):
        for g in combo:
            paths[appearances[g], g] = split_idx
            appearances[g] += 1
    return paths


def assemble_path_returns(
    split_returns: Sequence[pd.Series],
    n_samples: int,
    n_groups: int = 6,
    n_test_groups: int = 2,
) -> pd.DataFrame:
    """Stitch per-split out-of-sample returns into the φ full backtest paths.

    The missing piece between the CPCV machinery and PBO-style diagnostics:
    after running a model over :func:`combinatorial_purged_splits`, each
    split yields an out-of-sample return series on its test bars. This
    weaves those C(N, k) series into φ = C(N−1, k−1) *complete* return
    paths using the :func:`assemble_backtest_paths` assignment — the
    per-path performance distribution that
    :func:`~src.validation.pbo.probability_of_backtest_overfitting` and
    plain path-Sharpe histograms consume.

    Args:
        split_returns: One Series per split, in the split generator's
            order, indexed by *integer bar position* and covering at least
            that split's test indices (extra bars are ignored).
        n_samples: Number of ordered observations.
        n_groups: Contiguous blocks N used for the splits.
        n_test_groups: Test blocks per split k.

    Returns:
        DataFrame of shape ``(n_samples, φ)``: column ``j`` is backtest
        path ``j``'s return per bar, every bar covered exactly once.

    Raises:
        ValueError: If the split count is wrong, a split is missing test
            bars, or values are NaN.
    """
    _validate_groups(n_groups, n_test_groups)
    expected = math.comb(n_groups, n_test_groups)
    if len(split_returns) != expected:
        raise ValueError(f"expected {expected} split series, got {len(split_returns)}.")

    blocks = np.array_split(np.arange(n_samples), n_groups)
    paths = assemble_backtest_paths(n_groups=n_groups, n_test_groups=n_test_groups)
    phi = paths.shape[0]

    out = np.empty((n_samples, phi))
    for j in range(phi):
        for g in range(n_groups):
            series = split_returns[paths[j, g]]
            needed = blocks[g]
            missing = needed[~np.isin(needed, series.index)]
            if len(missing):
                raise ValueError(
                    f"split {paths[j, g]} is missing test bars {missing[:5].tolist()}"
                    f" required for group {g}."
                )
            values = series.loc[needed].to_numpy(dtype=float)
            if np.isnan(values).any():
                raise ValueError(f"split {paths[j, g]} contains NaN returns.")
            out[needed, j] = values

    return pd.DataFrame(out, index=np.arange(n_samples), columns=list(range(phi)))


def _validate_groups(n_groups: int, n_test_groups: int) -> None:
    """Shared range checks for the CPCV group parameters."""
    if n_groups < 2:
        raise ValueError(f"n_groups must be >= 2, got {n_groups}.")
    if not 1 <= n_test_groups < n_groups:
        raise ValueError(
            f"n_test_groups must be in [1, n_groups), got {n_test_groups} for {n_groups} groups."
        )
