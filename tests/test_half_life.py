"""Tests for the Ornstein-Uhlenbeck half-life estimator."""

import math

import numpy as np
import pandas as pd
import pytest

from src.strategy import fit_ou, ou_half_life


def _deterministic_ar1(phi: float, mu: float, s0: float, n: int = 200) -> pd.Series:
    """Noiseless AR(1): S_t = mu + phi (S_{t-1} - mu)."""
    s = [s0]
    for _ in range(n - 1):
        s.append(mu + phi * (s[-1] - mu))
    return pd.Series(s)


def _stochastic_ou(theta: float, mu: float, n: int = 5000, seed: int = 0) -> pd.Series:
    rng = np.random.default_rng(seed)
    s = np.zeros(n)
    s[0] = mu
    for t in range(1, n):
        s[t] = s[t - 1] + theta * (mu - s[t - 1]) + rng.normal(0.0, 1.0)
    return pd.Series(s)


def test_recovers_exact_ar1_parameters() -> None:
    fit = fit_ou(_deterministic_ar1(phi=0.5, mu=10.0, s0=20.0))
    assert fit.phi == pytest.approx(0.5)
    assert fit.mu == pytest.approx(10.0)
    assert fit.theta == pytest.approx(math.log(2))
    assert fit.half_life == pytest.approx(1.0)


def test_stochastic_ou_half_life_near_truth() -> None:
    # theta=0.1 -> true half-life = ln2 / 0.1 = 6.93 bars
    fit = fit_ou(_stochastic_ou(theta=0.1, mu=50.0))
    assert fit.half_life == pytest.approx(math.log(2) / 0.1, rel=0.15)
    assert fit.mu == pytest.approx(50.0, abs=1.0)


def test_faster_reversion_gives_shorter_half_life() -> None:
    slow = ou_half_life(_stochastic_ou(theta=0.1, mu=50.0, seed=1))
    fast = ou_half_life(_stochastic_ou(theta=0.5, mu=50.0, seed=1))
    assert fast < slow


def test_random_walk_has_a_very_long_half_life() -> None:
    rng = np.random.default_rng(2)
    rw = pd.Series(100.0 + np.cumsum(rng.normal(0.0, 1.0, 3000)))
    mean_reverter = ou_half_life(_stochastic_ou(theta=0.2, mu=50.0, seed=2))
    # a random walk barely reverts (phi ~ 1): its half-life dwarfs a real
    # mean-reverting spread's
    assert ou_half_life(rw) > 10 * mean_reverter


def test_explosive_series_is_not_mean_reverting() -> None:
    explosive = pd.Series([100.0 * 1.05**i for i in range(50)])
    fit = fit_ou(explosive)
    assert fit.phi > 1.0
    assert fit.half_life == math.inf


def test_wrapper_matches_fit() -> None:
    s = _stochastic_ou(theta=0.15, mu=20.0, seed=3)
    assert ou_half_life(s) == fit_ou(s).half_life


def test_bad_inputs_raise() -> None:
    with pytest.raises(ValueError, match="at least 3"):
        fit_ou(pd.Series([1.0, 2.0]))
    with pytest.raises(ValueError, match="NaN"):
        fit_ou(pd.Series([1.0, np.nan, 3.0, 4.0]))
    with pytest.raises(ValueError, match="constant"):
        fit_ou(pd.Series([5.0, 5.0, 5.0, 5.0]))
