"""Tests for stitching CPCV split returns into full backtest paths."""

import numpy as np
import pandas as pd
import pytest

from src.validation import (
    assemble_path_returns,
    combinatorial_purged_splits,
    n_backtest_paths,
)


def _perfect_split_returns(full: np.ndarray, n_groups: int, n_test_groups: int) -> list[pd.Series]:
    """Each split reports the TRUE returns on its own test bars."""
    splits = combinatorial_purged_splits(len(full), n_groups, n_test_groups)
    return [pd.Series(full[test], index=test) for _, test in splits]


def test_identity_when_splits_report_the_true_series() -> None:
    rng = np.random.default_rng(4)
    full = rng.normal(0.0005, 0.01, 120)
    split_returns = _perfect_split_returns(full, n_groups=6, n_test_groups=2)
    paths = assemble_path_returns(split_returns, 120, n_groups=6, n_test_groups=2)
    assert paths.shape == (120, n_backtest_paths(6, 2))
    # a model that reproduces reality gives every path = the true series
    for j in paths.columns:
        assert np.allclose(paths[j].to_numpy(), full)


def test_split_dependent_returns_produce_distinct_paths() -> None:
    n, groups, k = 90, 6, 2
    splits = combinatorial_purged_splits(n, groups, k)
    # each split reports a CONSTANT equal to its own index -> paths differ
    split_returns = [pd.Series(float(i), index=test) for i, (_, test) in enumerate(splits)]
    paths = assemble_path_returns(split_returns, n, groups, k)
    assert paths.shape[1] == n_backtest_paths(groups, k)
    assert len({tuple(paths[j]) for j in paths.columns}) == paths.shape[1]


def test_every_bar_is_covered_in_every_path() -> None:
    rng = np.random.default_rng(1)
    full = rng.normal(0, 0.01, 60)
    split_returns = _perfect_split_returns(full, 5, 2)
    paths = assemble_path_returns(split_returns, 60, 5, 2)
    assert not paths.isna().any().any()


def test_path_sharpe_distribution_use_case() -> None:
    rng = np.random.default_rng(2)
    full = rng.normal(0.001, 0.01, 240)
    split_returns = _perfect_split_returns(full, 6, 2)
    paths = assemble_path_returns(split_returns, 240, 6, 2)
    sharpes = paths.mean() / paths.std(ddof=1) * np.sqrt(252)
    assert len(sharpes) == 5
    assert np.isfinite(sharpes.to_numpy()).all()


def test_extra_bars_in_a_split_are_ignored() -> None:
    rng = np.random.default_rng(3)
    full = rng.normal(0, 0.01, 60)
    splits = combinatorial_purged_splits(60, 5, 2)
    # splits report returns on ALL bars, not just their test bars
    split_returns = [pd.Series(full, index=np.arange(60)) for _ in splits]
    paths = assemble_path_returns(split_returns, 60, 5, 2)
    assert np.allclose(paths[0].to_numpy(), full)


def test_wrong_split_count_raises() -> None:
    with pytest.raises(ValueError, match="expected 15"):
        assemble_path_returns([pd.Series(dtype=float)], 60, 6, 2)


def test_missing_test_bars_raise() -> None:
    splits = combinatorial_purged_splits(60, 5, 2)
    split_returns = [pd.Series(0.0, index=test[:-2]) for _, test in splits]  # truncated
    with pytest.raises(ValueError, match="missing test bars"):
        assemble_path_returns(split_returns, 60, 5, 2)


def test_nan_returns_raise() -> None:
    splits = combinatorial_purged_splits(60, 5, 2)
    split_returns = [pd.Series(0.0, index=test) for _, test in splits]
    split_returns[0].iloc[3] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        assemble_path_returns(split_returns, 60, 5, 2)
