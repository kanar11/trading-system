"""Tests for CPCV backtest-path assembly."""

import math
from itertools import combinations

import numpy as np
import pytest

from src.validation import (
    assemble_backtest_paths,
    combinatorial_purged_splits,
    n_backtest_paths,
)


def test_shape_is_phi_by_n_groups() -> None:
    paths = assemble_backtest_paths(n_groups=6, n_test_groups=2)
    assert paths.shape == (n_backtest_paths(6, 2), 6)
    assert paths.shape == (5, 6)


def test_every_assigned_split_actually_tests_the_group() -> None:
    n, k = 6, 2
    combos = list(combinations(range(n), k))
    paths = assemble_backtest_paths(n_groups=n, n_test_groups=k)
    for j in range(paths.shape[0]):
        for g in range(n):
            assert g in combos[paths[j, g]]


def test_each_split_group_pair_used_exactly_once() -> None:
    n, k = 6, 2
    paths = assemble_backtest_paths(n_groups=n, n_test_groups=k)
    pairs = {(int(paths[j, g]), g) for j in range(paths.shape[0]) for g in range(n)}
    # phi * N distinct pairs = k forecasts consumed from each of C(N,k) splits
    assert len(pairs) == paths.size
    assert len(pairs) == math.comb(n, k) * k


def test_single_test_group_gives_one_kfold_path() -> None:
    paths = assemble_backtest_paths(n_groups=5, n_test_groups=1)
    # k=1 is plain purged K-fold: one path, group g tested by split g
    assert paths.shape == (1, 5)
    assert list(paths[0]) == [0, 1, 2, 3, 4]


def test_first_path_takes_first_appearances() -> None:
    # combinations order for (6,2): (0,1),(0,2),(0,3),(0,4),(0,5),(1,2),...
    paths = assemble_backtest_paths(n_groups=6, n_test_groups=2)
    assert list(paths[0]) == [0, 0, 1, 2, 3, 4]


def test_paths_compose_with_the_split_generator() -> None:
    n_samples, n, k = 90, 6, 2
    splits = combinatorial_purged_splits(n_samples, n_groups=n, n_test_groups=k)
    blocks = np.array_split(np.arange(n_samples), n)
    paths = assemble_backtest_paths(n_groups=n, n_test_groups=k)
    for j in range(paths.shape[0]):
        covered: list[np.ndarray] = []
        for g in range(n):
            _, test_idx = splits[paths[j, g]]
            assert np.isin(blocks[g], test_idx).all()  # split tests block g
            covered.append(blocks[g])
        # each path covers the full sample exactly once
        stitched = np.sort(np.concatenate(covered))
        assert np.array_equal(stitched, np.arange(n_samples))


def test_bad_group_counts_raise() -> None:
    with pytest.raises(ValueError, match="n_test_groups"):
        assemble_backtest_paths(n_groups=4, n_test_groups=4)
    with pytest.raises(ValueError, match="n_groups"):
        assemble_backtest_paths(n_groups=1, n_test_groups=1)
