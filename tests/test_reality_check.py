"""Tests for White's Reality Check."""

import numpy as np
import pytest

from src.validation.reality_check import whites_reality_check


def test_dominant_strategy_is_significant() -> None:
    rng = np.random.default_rng(0)
    noise = rng.normal(0.0, 0.01, size=(250, 9))
    winner = rng.normal(0.003, 0.01, size=(250, 1))  # strong real edge
    panel = np.hstack([winner, noise])
    result = whites_reality_check(panel, n_bootstrap=1000, seed=1)
    assert result.best_strategy == 0
    assert result.p_value < 0.05


def test_no_spurious_significance_for_identical_noise() -> None:
    rng = np.random.default_rng(2)
    col = rng.normal(0.0, 0.01, size=250)
    panel = np.tile(col.reshape(-1, 1), (1, 5))  # 5 identical strategies
    result = whites_reality_check(panel, n_bootstrap=1000, seed=3)
    assert result.p_value > 0.1


def test_p_value_in_unit_interval() -> None:
    rng = np.random.default_rng(4)
    panel = rng.normal(0.0, 0.01, size=(200, 8))
    result = whites_reality_check(panel, n_bootstrap=500, block_size=5, seed=5)
    assert 0.0 <= result.p_value <= 1.0


def test_deterministic() -> None:
    rng = np.random.default_rng(6)
    panel = rng.normal(0.0005, 0.01, size=(200, 6))
    a = whites_reality_check(panel, n_bootstrap=500, seed=7)
    b = whites_reality_check(panel, n_bootstrap=500, seed=7)
    assert a.p_value == b.p_value
    assert a.best_strategy == b.best_strategy


def test_validation_errors() -> None:
    with pytest.raises(ValueError, match="2-D"):
        whites_reality_check(np.zeros(100))
    with pytest.raises(ValueError, match="n_bootstrap"):
        whites_reality_check(np.zeros((100, 3)), n_bootstrap=0)
    with pytest.raises(ValueError, match="block_size"):
        whites_reality_check(np.zeros((100, 3)), block_size=200)
