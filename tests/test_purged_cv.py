"""Tests for purged & embargoed K-fold cross-validation splits."""

import numpy as np
import pytest

from src.validation.purged_cv import purged_kfold_splits


def test_test_folds_partition_all_samples() -> None:
    splits = purged_kfold_splits(100, n_splits=5)
    assert len(splits) == 5
    all_test = np.concatenate([test for _, test in splits])
    assert sorted(all_test.tolist()) == list(range(100))


def test_plain_kfold_train_is_complement() -> None:
    for train, test in purged_kfold_splits(100, n_splits=5):
        assert len(train) + len(test) == 100
        assert not (set(train.tolist()) & set(test.tolist()))


def test_purge_removes_bars_before_test() -> None:
    train, test = purged_kfold_splits(100, n_splits=5, purge=3)[2]  # test 40-59
    assert set(range(37, 40)).isdisjoint(train.tolist())
    assert set(range(37, 40)).isdisjoint(test.tolist())


def test_embargo_removes_bars_after_test() -> None:
    train, test = purged_kfold_splits(100, n_splits=5, embargo=0.05)[2]  # test 40-59, end 60
    assert set(range(60, 65)).isdisjoint(train.tolist())  # 5% of 100 = 5 bars


def test_train_never_overlaps_test() -> None:
    for train, test in purged_kfold_splits(120, n_splits=6, embargo=0.02, purge=2):
        assert set(train.tolist()).isdisjoint(test.tolist())


def test_first_fold_has_no_purge_underflow() -> None:
    train, test = purged_kfold_splits(50, n_splits=5, purge=5)[0]
    assert test[0] == 0
    assert len(train) > 0


def test_validation_errors() -> None:
    with pytest.raises(ValueError, match="n_splits must be >= 2"):
        purged_kfold_splits(100, n_splits=1)
    with pytest.raises(ValueError, match="cannot exceed"):
        purged_kfold_splits(3, n_splits=5)
    with pytest.raises(ValueError, match="embargo"):
        purged_kfold_splits(100, n_splits=5, embargo=-0.1)
    with pytest.raises(ValueError, match="purge"):
        purged_kfold_splits(100, n_splits=5, purge=-1)
