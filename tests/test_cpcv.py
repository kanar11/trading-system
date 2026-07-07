"""Tests for Combinatorial Purged Cross-Validation."""

import math

import numpy as np
import pytest

from src.validation import combinatorial_purged_splits, n_backtest_paths


def test_number_of_splits_is_n_choose_k() -> None:
    splits = combinatorial_purged_splits(120, n_groups=6, n_test_groups=2)
    assert len(splits) == math.comb(6, 2)  # 15


def test_train_and_test_are_disjoint() -> None:
    for train, test in combinatorial_purged_splits(100, n_groups=5, n_test_groups=2):
        assert len(np.intersect1d(train, test)) == 0


def test_no_purge_no_embargo_covers_everything() -> None:
    for train, test in combinatorial_purged_splits(90, n_groups=6, n_test_groups=2):
        combined = np.sort(np.concatenate([train, test]))
        assert np.array_equal(combined, np.arange(90))


def test_each_sample_tested_phi_times() -> None:
    # every group sits in C(N-1, k-1) combinations, so each sample appears
    # in exactly n_backtest_paths test sets
    n, groups, k = 60, 6, 2
    counts = np.zeros(n, dtype=int)
    for _, test in combinatorial_purged_splits(n, n_groups=groups, n_test_groups=k):
        counts[test] += 1
    assert (counts == n_backtest_paths(groups, k)).all()


def test_purge_removes_bars_before_each_test_block() -> None:
    splits = combinatorial_purged_splits(100, n_groups=5, n_test_groups=1, purge=3)
    # second fold: test = [20, 40); bars 17-19 must be purged from training
    train, test = splits[1]
    assert test[0] == 20
    for bar in (17, 18, 19):
        assert bar not in train
    assert 16 in train


def test_embargo_removes_bars_after_each_test_block() -> None:
    splits = combinatorial_purged_splits(100, n_groups=5, n_test_groups=1, embargo=0.05)
    # second fold: test = [20, 40); embargo of 5 bars removes 40-44
    train, test = splits[1]
    assert test[-1] == 39
    for bar in (40, 41, 42, 43, 44):
        assert bar not in train
    assert 45 in train


def test_adjacent_test_blocks_purge_correctly() -> None:
    # blocks 0 and 1 adjacent: purge before block 1 lies inside block 0 (already
    # test) and must not corrupt anything
    splits = combinatorial_purged_splits(100, n_groups=5, n_test_groups=2, purge=5)
    train, test = splits[0]  # combo (0, 1) -> test = [0, 40)
    assert np.array_equal(test, np.arange(40))
    assert len(np.intersect1d(train, test)) == 0
    assert 40 in train  # nothing purged after the last test bar


def test_n_backtest_paths_values() -> None:
    assert n_backtest_paths(6, 2) == 5
    assert n_backtest_paths(10, 2) == 9
    assert n_backtest_paths(5, 1) == 1  # plain purged K-fold: one path


def test_bad_group_counts_raise() -> None:
    with pytest.raises(ValueError, match="n_test_groups"):
        combinatorial_purged_splits(100, n_groups=5, n_test_groups=5)
    with pytest.raises(ValueError, match="n_groups"):
        combinatorial_purged_splits(100, n_groups=1, n_test_groups=1)
    with pytest.raises(ValueError, match="n_groups"):
        combinatorial_purged_splits(4, n_groups=6, n_test_groups=2)
    with pytest.raises(ValueError, match="n_test_groups"):
        n_backtest_paths(6, 0)


def test_negative_purge_or_embargo_raises() -> None:
    with pytest.raises(ValueError, match="purge"):
        combinatorial_purged_splits(100, purge=-1)
    with pytest.raises(ValueError, match="embargo"):
        combinatorial_purged_splits(100, embargo=-0.1)
