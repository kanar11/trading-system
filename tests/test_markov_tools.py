"""Tests for stationary distribution and regime probability forecasts."""

import numpy as np
import pandas as pd
import pytest

from src.regime import (
    forecast_regime_probabilities,
    regime_transition_matrix,
    stationary_distribution,
)


def _matrix(rows: list[list[float]], labels: list[str]) -> pd.DataFrame:
    return pd.DataFrame(rows, index=labels, columns=labels)


def test_symmetric_two_state_chain_is_fifty_fifty() -> None:
    matrix = _matrix([[0.9, 0.1], [0.1, 0.9]], ["bull", "bear"])
    pi = stationary_distribution(matrix)
    assert pi.name == "stationary"
    assert np.allclose(pi.to_numpy(), [0.5, 0.5])
    assert float(pi.sum()) == pytest.approx(1.0)


def test_asymmetric_chain_matches_closed_form() -> None:
    # two-state chain: pi = (b/(a+b), a/(a+b)) for leave-rates a, b
    a, b = 0.2, 0.05
    matrix = _matrix([[1 - a, a], [b, 1 - b]], ["calm", "wild"])
    pi = stationary_distribution(matrix)
    assert pi["calm"] == pytest.approx(b / (a + b))
    assert pi["wild"] == pytest.approx(a / (a + b))


def test_absorbing_state_takes_all_mass() -> None:
    matrix = _matrix([[0.8, 0.2], [0.0, 1.0]], ["transient", "absorbing"])
    pi = stationary_distribution(matrix)
    assert pi["absorbing"] == pytest.approx(1.0)
    assert pi["transient"] == pytest.approx(0.0, abs=1e-12)


def test_stationary_matches_empirical_frequencies() -> None:
    rng = np.random.default_rng(3)
    # simulate a persistent 2-state chain and compare pi to observed shares
    states = [0]
    for _ in range(20_000):
        stay = 0.95 if states[-1] == 0 else 0.90
        states.append(states[-1] if rng.random() < stay else 1 - states[-1])
    series = pd.Series(states)
    matrix = regime_transition_matrix(series)
    pi = stationary_distribution(matrix)
    observed = series.value_counts(normalize=True).sort_index()
    assert np.allclose(pi.to_numpy(), observed.to_numpy(), atol=0.02)


def test_forecast_step_zero_returns_start() -> None:
    matrix = _matrix([[0.9, 0.1], [0.3, 0.7]], ["a", "b"])
    out = forecast_regime_probabilities(matrix, "a", steps=0)
    assert list(out) == [1.0, 0.0]
    assert out.name == "forecast"


def test_forecast_one_step_is_the_matrix_row() -> None:
    matrix = _matrix([[0.9, 0.1], [0.3, 0.7]], ["a", "b"])
    out = forecast_regime_probabilities(matrix, "b", steps=1)
    assert np.allclose(out.to_numpy(), [0.3, 0.7])


def test_forecast_converges_to_stationary() -> None:
    matrix = _matrix([[0.9, 0.1], [0.3, 0.7]], ["a", "b"])
    far = forecast_regime_probabilities(matrix, "a", steps=200)
    pi = stationary_distribution(matrix)
    assert np.allclose(far.to_numpy(), pi.to_numpy(), atol=1e-9)


def test_forecast_accepts_a_start_distribution() -> None:
    matrix = _matrix([[0.9, 0.1], [0.3, 0.7]], ["a", "b"])
    start = pd.Series({"a": 0.5, "b": 0.5})
    out = forecast_regime_probabilities(matrix, start, steps=1)
    assert np.allclose(out.to_numpy(), [0.5 * 0.9 + 0.5 * 0.3, 0.5 * 0.1 + 0.5 * 0.7])
    assert float(out.sum()) == pytest.approx(1.0)


def test_bad_inputs_raise() -> None:
    matrix = _matrix([[0.9, 0.1], [0.3, 0.7]], ["a", "b"])
    with pytest.raises(ValueError, match="sum to 1"):
        stationary_distribution(_matrix([[0.5, 0.4], [0.3, 0.7]], ["a", "b"]))
    with pytest.raises(ValueError, match="labels"):
        bad = pd.DataFrame([[1.0, 0.0], [0.0, 1.0]], index=["a", "b"], columns=["x", "y"])
        stationary_distribution(bad)
    with pytest.raises(ValueError, match="non-negative"):
        stationary_distribution(_matrix([[1.1, -0.1], [0.0, 1.0]], ["a", "b"]))
    with pytest.raises(ValueError, match="steps"):
        forecast_regime_probabilities(matrix, "a", steps=-1)
    with pytest.raises(ValueError, match="unknown regime"):
        forecast_regime_probabilities(matrix, "zzz")
    with pytest.raises(ValueError, match="probability vector"):
        forecast_regime_probabilities(matrix, pd.Series({"a": 0.9, "b": 0.9}))
