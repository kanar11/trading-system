"""Tests for the CSCV probability-of-backtest-overfitting estimator."""

from math import comb

import numpy as np
import pytest

from src.validation.pbo import probability_of_backtest_overfitting


def test_pbo_in_unit_interval_and_shapes() -> None:
    rng = np.random.default_rng(0)
    m = rng.normal(0, 0.01, size=(500, 12))
    result = probability_of_backtest_overfitting(m, n_blocks=10)
    assert 0.0 <= result.pbo <= 1.0
    assert result.n_strategies == 12
    assert result.n_combinations == comb(10, 5)
    assert result.logits.shape == (comb(10, 5),)
    assert result.is_best_oos_relative_rank.shape == (comb(10, 5),)


def test_dominant_strategy_has_low_pbo() -> None:
    rng = np.random.default_rng(1)
    noise = rng.normal(0.0, 0.01, size=(600, 9))
    winner = rng.normal(0.003, 0.01, size=(600, 1))  # genuine, persistent edge
    m = np.hstack([winner, noise])
    result = probability_of_backtest_overfitting(m, n_blocks=10)
    assert result.pbo < 0.3


def test_pure_noise_pbo_near_half() -> None:
    rng = np.random.default_rng(2)
    m = rng.normal(0, 0.01, size=(600, 20))
    result = probability_of_backtest_overfitting(m, n_blocks=10)
    assert 0.25 <= result.pbo <= 0.75


def test_systematic_overfit_gives_high_pbo() -> None:
    # config 0 dominates block A but is worst in block B, and vice versa:
    # whichever is IS-best is OOS-worst on every split -> PBO == 1.
    m = np.array(
        [
            [0.02, -0.02],
            [0.03, -0.01],
            [-0.02, 0.02],
            [-0.01, 0.03],
        ]
    )
    result = probability_of_backtest_overfitting(m, n_blocks=2)
    assert result.pbo == 1.0
    assert result.n_combinations == 2


def test_requires_two_configs() -> None:
    with pytest.raises(ValueError, match="2 configurations"):
        probability_of_backtest_overfitting(np.zeros((100, 1)), n_blocks=4)


def test_n_blocks_must_be_even() -> None:
    with pytest.raises(ValueError, match="even"):
        probability_of_backtest_overfitting(np.zeros((100, 3)), n_blocks=5)


def test_requires_enough_observations() -> None:
    with pytest.raises(ValueError, match="observations"):
        probability_of_backtest_overfitting(np.zeros((10, 3)), n_blocks=10)


def test_rejects_non_2d() -> None:
    with pytest.raises(ValueError, match="2-D"):
        probability_of_backtest_overfitting(np.zeros(100), n_blocks=4)
