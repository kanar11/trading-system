"""Tests for Hansen's Superior Predictive Ability test."""

import numpy as np
import pytest

from src.validation import hansen_spa, whites_reality_check


def _noise_panel(n: int = 300, k: int = 8, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(0.0, 0.01, size=(n, k))


def test_genuine_winner_gets_small_p_value() -> None:
    panel = _noise_panel()
    panel[:, 3] += 0.004  # a real edge, ~0.4 sigma per period
    result = hansen_spa(panel, n_bootstrap=500, seed=1)
    assert result.best_strategy == 3
    assert result.p_value < 0.05
    assert result.test_statistic > 0


def test_pure_noise_is_not_significant() -> None:
    result = hansen_spa(_noise_panel(seed=20), n_bootstrap=500, seed=2)
    assert result.p_value > 0.10


def test_p_values_are_ordered_lower_consistent_upper() -> None:
    panel = _noise_panel(seed=7)
    panel[:, 0] += 0.002
    panel[:, 4] -= 0.02  # a deeply losing strategy
    result = hansen_spa(panel, n_bootstrap=500, seed=3)
    assert result.p_value_lower <= result.p_value <= result.p_value_upper


def test_studentisation_prefers_high_t_stat_over_raw_mean() -> None:
    rng = np.random.default_rng(3)
    n = 400
    steady = rng.normal(0.001, 0.001, n)  # small mean, tiny vol -> huge t
    wild = rng.normal(0.002, 0.06, n)  # bigger sample mean, huge vol -> tiny t
    panel = np.column_stack([steady, wild])
    spa = hansen_spa(panel, n_bootstrap=300, seed=4)
    rc = whites_reality_check(panel, n_bootstrap=300, seed=4)
    assert spa.best_strategy == 0  # studentised pick
    assert rc.best_strategy == 1  # raw-mean pick (the RC weakness)


def test_terrible_strategies_do_not_rescue_the_null() -> None:
    # the consistent p-value must stay small when garbage candidates are
    # appended — the SPA improvement over the RC recentring
    panel = _noise_panel(n=300, k=3, seed=9)
    panel[:, 0] += 0.004
    garbage = np.full((300, 5), -0.05) + _noise_panel(n=300, k=5, seed=10)
    stuffed = np.column_stack([panel, garbage])
    result = hansen_spa(stuffed, n_bootstrap=500, seed=5)
    assert result.best_strategy == 0
    assert result.p_value < 0.05


def test_zero_variance_column_is_ignored() -> None:
    panel = _noise_panel(n=200, k=2, seed=13)
    panel[:, 0] += 0.003
    stuffed = np.column_stack([panel, np.zeros(200)])  # constant column
    result = hansen_spa(stuffed, n_bootstrap=300, seed=6)
    assert result.best_strategy == 0
    assert np.isfinite(result.test_statistic)


def test_all_constant_panel_raises() -> None:
    with pytest.raises(ValueError, match="zero variance"):
        hansen_spa(np.ones((50, 3)), n_bootstrap=100, seed=0)


def test_seed_makes_results_reproducible() -> None:
    panel = _noise_panel(seed=15)
    a = hansen_spa(panel, n_bootstrap=200, seed=42)
    b = hansen_spa(panel, n_bootstrap=200, seed=42)
    assert a.p_value == b.p_value
    assert a.p_value_lower == b.p_value_lower
    assert a.p_value_upper == b.p_value_upper


def test_bad_inputs_raise() -> None:
    panel = _noise_panel(n=20, k=2)
    with pytest.raises(ValueError, match="2-D"):
        hansen_spa(panel[:, 0])
    with pytest.raises(ValueError, match="3 periods"):
        hansen_spa(panel[:2])
    with pytest.raises(ValueError, match="n_bootstrap"):
        hansen_spa(panel, n_bootstrap=0)
    with pytest.raises(ValueError, match="block_size"):
        hansen_spa(panel, block_size=0)
    with pytest.raises(ValueError, match="block_size"):
        hansen_spa(panel, block_size=21)
