"""Purged & embargoed K-fold cross-validation for time series.

Plain K-fold leaks information in a backtest: a training observation adjacent to
the test fold shares (almost) the same market state and, for multi-bar labels,
the same outcome window. López de Prado's fix (Advances in Financial Machine
Learning, 2018) is to **purge** training observations whose label window
overlaps the test fold and to **embargo** a span of training observations
immediately after it.

This module yields leakage-safe train/test index splits over a series of
``n_samples`` ordered observations. Pure numpy; index-based, so it composes with
any model or backtest.
"""

from __future__ import annotations

import numpy as np


def purged_kfold_splits(
    n_samples: int,
    n_splits: int = 5,
    embargo: float = 0.0,
    purge: int = 0,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Generate purged & embargoed K-fold train/test index splits.

    Test folds are contiguous blocks that partition ``[0, n_samples)``. For each
    fold, training indices that fall in the ``purge`` bars immediately *before*
    the test block, or in the embargo span immediately *after* it, are removed.

    Args:
        n_samples: Number of ordered observations.
        n_splits: Number of folds (2 <= n_splits <= n_samples).
        embargo: Embargo span after each test fold, as a fraction of
            ``n_samples`` (e.g. 0.01 = 1%). Rounded up to whole bars.
        purge: Number of bars before each test fold to purge from training.

    Returns:
        A list of ``(train_idx, test_idx)`` numpy arrays, one per fold.

    Raises:
        ValueError: If ``n_splits`` is out of range or ``embargo`` / ``purge``
            is negative.
    """
    if n_splits < 2:
        raise ValueError(f"n_splits must be >= 2, got {n_splits}.")
    if n_splits > n_samples:
        raise ValueError(f"n_splits ({n_splits}) cannot exceed n_samples ({n_samples}).")
    if embargo < 0:
        raise ValueError(f"embargo must be >= 0, got {embargo}.")
    if purge < 0:
        raise ValueError(f"purge must be >= 0, got {purge}.")

    indices = np.arange(n_samples)
    blocks = np.array_split(indices, n_splits)
    embargo_size = int(np.ceil(embargo * n_samples))

    splits: list[tuple[np.ndarray, np.ndarray]] = []
    for block in blocks:
        start = int(block[0])
        end = int(block[-1]) + 1

        train_mask = np.ones(n_samples, dtype=bool)
        train_mask[start:end] = False  # the test fold itself
        train_mask[max(start - purge, 0) : start] = False  # purge before
        train_mask[end : min(end + embargo_size, n_samples)] = False  # embargo after

        splits.append((indices[train_mask], block))
    return splits
