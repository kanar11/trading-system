"""Tests for the Gaussian HMM regime detector."""

import numpy as np
import pandas as pd
import pytest

from src.regime.hmm import HMMConfig, detect_hmm_regime, fit_gaussian_hmm


def _two_regime_series(n: int = 300, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Calm (high-mean, low-vol) then turbulent (low-mean, high-vol) returns.

    Returns (observations, ground_truth) where ground truth uses the
    mean-sorted convention: 0 = turbulent (lower mean), 1 = calm (higher mean).
    """
    rng = np.random.default_rng(seed)
    calm = rng.normal(0.002, 0.004, n)
    turbulent = rng.normal(-0.003, 0.025, n)
    x = np.concatenate([calm, turbulent])
    truth = np.concatenate([np.ones(n, dtype=int), np.zeros(n, dtype=int)])
    return x, truth


def test_recovers_two_regimes() -> None:
    x, truth = _two_regime_series()
    result = fit_gaussian_hmm(x, HMMConfig(n_states=2))
    # permutation-invariant agreement: the model must separate the regimes
    agree = (result.states == truth).mean()
    agree = max(agree, 1.0 - agree)
    assert agree > 0.85


def test_states_sorted_by_mean() -> None:
    x, _ = _two_regime_series()
    result = fit_gaussian_hmm(x, HMMConfig(n_states=2))
    assert result.state_means[0] <= result.state_means[1]


def test_fit_is_deterministic() -> None:
    x, _ = _two_regime_series()
    a = fit_gaussian_hmm(x, HMMConfig(n_states=2))
    b = fit_gaussian_hmm(x, HMMConfig(n_states=2))
    assert np.array_equal(a.states, b.states)
    assert a.log_likelihood == pytest.approx(b.log_likelihood)
    assert np.allclose(a.state_means, b.state_means)


def test_log_likelihood_improves_with_iterations() -> None:
    x, _ = _two_regime_series()
    one = fit_gaussian_hmm(x, HMMConfig(n_states=2, max_iter=1))
    many = fit_gaussian_hmm(x, HMMConfig(n_states=2, max_iter=50))
    # Baum-Welch never decreases the data log-likelihood
    assert many.log_likelihood >= one.log_likelihood - 1e-9


def test_transition_rows_sum_to_one() -> None:
    x, _ = _two_regime_series()
    result = fit_gaussian_hmm(x, HMMConfig(n_states=2))
    assert np.allclose(result.transition.sum(axis=1), 1.0)


def test_posterior_is_valid_distribution() -> None:
    x, _ = _two_regime_series()
    result = fit_gaussian_hmm(x, HMMConfig(n_states=2))
    assert result.posterior.shape == (len(x), 2)
    assert np.allclose(result.posterior.sum(axis=1), 1.0)
    assert (result.posterior >= 0).all()


def test_single_state() -> None:
    rng = np.random.default_rng(3)
    x = rng.normal(0, 0.01, 100)
    result = fit_gaussian_hmm(x, HMMConfig(n_states=1))
    assert set(result.states.tolist()) == {0}
    assert result.transition.shape == (1, 1)
    assert result.transition[0, 0] == pytest.approx(1.0)


def test_too_few_observations_raises() -> None:
    with pytest.raises(ValueError, match="at least"):
        fit_gaussian_hmm(np.array([0.01, -0.02, 0.0]), HMMConfig(n_states=2))


def test_invalid_n_states_raises() -> None:
    with pytest.raises(ValueError, match="n_states"):
        fit_gaussian_hmm(np.zeros(50), HMMConfig(n_states=0))


def test_constant_series_does_not_crash() -> None:
    result = fit_gaussian_hmm(np.full(100, 0.001), HMMConfig(n_states=2))
    assert result.states.shape == (100,)
    assert result.state_means.shape == (2,)


def test_detect_hmm_regime_aligns_and_drops_na() -> None:
    rng = np.random.default_rng(1)
    idx = pd.date_range("2020-01-01", periods=200, freq="B")
    s = pd.Series(rng.normal(0, 0.01, 200), index=idx)
    s.iloc[5] = np.nan
    out = detect_hmm_regime(s, n_states=2)
    assert out.name == "hmm_state"
    assert len(out) == 199
    assert out.index.equals(s.dropna().index)
    assert set(out.unique()).issubset({0, 1})
