"""Tests for Markov entropy rate and regime predictability."""

import math

import pandas as pd
import pytest

from src.regime import markov_entropy_rate, regime_predictability


def _matrix(rows: list[list[float]], labels: list[str]) -> pd.DataFrame:
    return pd.DataFrame(rows, index=labels, columns=labels)


def test_deterministic_chain_has_zero_entropy() -> None:
    cycle = _matrix([[0.0, 1.0], [1.0, 0.0]], ["a", "b"])
    assert markov_entropy_rate(cycle) == pytest.approx(0.0)
    assert regime_predictability(cycle) == pytest.approx(1.0)


def test_uniform_chain_is_pure_noise() -> None:
    uniform = _matrix([[0.5, 0.5], [0.5, 0.5]], ["a", "b"])
    assert markov_entropy_rate(uniform) == pytest.approx(1.0)  # 1 bit
    assert regime_predictability(uniform) == pytest.approx(0.0)
    three = _matrix([[1 / 3] * 3] * 3, ["a", "b", "c"])
    assert markov_entropy_rate(three) == pytest.approx(math.log2(3))


def test_symmetric_sticky_chain_matches_binary_entropy() -> None:
    p = 0.9
    sticky = _matrix([[p, 1 - p], [1 - p, p]], ["a", "b"])
    binary_entropy = -(p * math.log2(p) + (1 - p) * math.log2(1 - p))
    # symmetric chain: stationary 50/50, both rows identical entropy
    assert markov_entropy_rate(sticky) == pytest.approx(binary_entropy)


def test_stickier_chains_are_more_predictable() -> None:
    loose = _matrix([[0.6, 0.4], [0.4, 0.6]], ["a", "b"])
    sticky = _matrix([[0.99, 0.01], [0.01, 0.99]], ["a", "b"])
    assert regime_predictability(sticky) > regime_predictability(loose)
    assert markov_entropy_rate(sticky) < markov_entropy_rate(loose)


def test_natural_log_base_gives_nats() -> None:
    uniform = _matrix([[0.5, 0.5], [0.5, 0.5]], ["a", "b"])
    nats = markov_entropy_rate(uniform, base=math.e)
    assert nats == pytest.approx(math.log(2))


def test_single_state_chain_is_trivially_predictable() -> None:
    single = _matrix([[1.0]], ["only"])
    assert markov_entropy_rate(single) == pytest.approx(0.0)
    assert regime_predictability(single) == pytest.approx(1.0)


def test_stationary_weighting_matters() -> None:
    # state 'a' is noisy but rarely visited; the entropy rate must weight
    # rows by the stationary distribution, not average them equally
    matrix = _matrix([[0.5, 0.5], [0.02, 0.98]], ["a", "b"])
    rate = markov_entropy_rate(matrix)
    equal_weight = 0.5 * 1.0 + 0.5 * (-(0.02 * math.log2(0.02) + 0.98 * math.log2(0.98)))
    assert rate < equal_weight  # pi(a) << 0.5 pulls the rate down


def test_bad_inputs_raise() -> None:
    matrix = _matrix([[0.5, 0.5], [0.5, 0.5]], ["a", "b"])
    with pytest.raises(ValueError, match="base"):
        markov_entropy_rate(matrix, base=1.0)
    with pytest.raises(ValueError, match="sum to 1"):
        markov_entropy_rate(_matrix([[0.5, 0.4], [0.5, 0.5]], ["a", "b"]))
    with pytest.raises(ValueError, match="sum to 1"):
        regime_predictability(_matrix([[0.5, 0.4], [0.5, 0.5]], ["a", "b"]))
