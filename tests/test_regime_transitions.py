"""Tests for regime transition analytics."""

import numpy as np
import pandas as pd
import pytest

from src.regime.transitions import regime_durations, regime_transition_matrix


def test_transition_matrix_values() -> None:
    # 0->0, 0->1, 1->1, 1->0
    m = regime_transition_matrix(pd.Series([0, 0, 1, 1, 0]))
    assert list(m.index) == [0, 1]
    assert list(m.columns) == [0, 1]
    assert m.loc[0, 0] == pytest.approx(0.5)
    assert m.loc[0, 1] == pytest.approx(0.5)
    assert m.loc[1, 1] == pytest.approx(0.5)
    assert m.loc[1, 0] == pytest.approx(0.5)


def test_transition_rows_sum_to_one() -> None:
    m = regime_transition_matrix(pd.Series([0, 1, 2, 0, 1, 2, 2, 0]))
    assert np.allclose(m.sum(axis=1).to_numpy(), 1.0)


def test_single_regime_self_transition() -> None:
    m = regime_transition_matrix(pd.Series([1, 1, 1]))
    assert m.loc[1, 1] == pytest.approx(1.0)


def test_transition_matrix_empty_for_short_input() -> None:
    assert regime_transition_matrix(pd.Series([0])).empty


def test_durations_mean_run_length() -> None:
    # runs: 0(len2), 1(len3), 0(len1) -> label 0 mean 1.5, label 1 mean 3
    d = regime_durations(pd.Series([0, 0, 1, 1, 1, 0]))
    assert d.loc[0] == pytest.approx(1.5)
    assert d.loc[1] == pytest.approx(3.0)


def test_durations_empty() -> None:
    assert regime_durations(pd.Series([], dtype=float)).empty
