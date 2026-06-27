"""Tests for participation-rate and Almgren-Chriss execution models."""

import numpy as np
import pytest

from src.execution.impact import (
    almgren_chriss_cost,
    almgren_chriss_trajectory,
    participation_rate_cost,
)

# --- participation_rate_cost ----------------------------------------------


def test_participation_cost_linear() -> None:
    assert participation_rate_cost(100, 1000, eta=0.1, exponent=1.0) == pytest.approx(0.01)


def test_participation_cost_doubles_with_size() -> None:
    base = participation_rate_cost(100, 1000, eta=0.1, exponent=1.0)
    assert participation_rate_cost(200, 1000, eta=0.1, exponent=1.0) == pytest.approx(2 * base)


def test_participation_cost_square_root_law() -> None:
    assert participation_rate_cost(400, 100, eta=1.0, exponent=0.5) == pytest.approx(2.0)


def test_participation_cost_sign_insensitive() -> None:
    assert participation_rate_cost(-100, 1000) == participation_rate_cost(100, 1000)


def test_participation_cost_guards() -> None:
    assert participation_rate_cost(0, 1000) == 0.0
    assert participation_rate_cost(100, 0) == 0.0


# --- almgren_chriss_trajectory --------------------------------------------


def test_trajectory_endpoints_and_length() -> None:
    traj = almgren_chriss_trajectory(1000.0, 10, urgency=0.5)
    assert len(traj) == 11
    assert traj[0] == pytest.approx(1000.0)
    assert traj[-1] == pytest.approx(0.0)


def test_trajectory_twap_is_linear() -> None:
    traj = almgren_chriss_trajectory(1000.0, 10, urgency=0.0)
    expected = 1000.0 * (10 - np.arange(11)) / 10
    assert np.allclose(traj, expected)


def test_trajectory_monotonic_decreasing() -> None:
    traj = almgren_chriss_trajectory(1000.0, 20, urgency=0.8)
    assert (np.diff(traj) <= 1e-9).all()


def test_higher_urgency_front_loads() -> None:
    slow = almgren_chriss_trajectory(1000.0, 20, urgency=0.1)
    fast = almgren_chriss_trajectory(1000.0, 20, urgency=1.5)
    # more urgent schedule has sold more by the midpoint -> less remaining
    assert fast[10] < slow[10]


def test_trajectory_validation() -> None:
    with pytest.raises(ValueError, match="n_steps"):
        almgren_chriss_trajectory(1000.0, 0)
    with pytest.raises(ValueError, match="urgency"):
        almgren_chriss_trajectory(1000.0, 10, urgency=-1.0)


# --- almgren_chriss_cost ---------------------------------------------------


def test_twap_cost_matches_closed_form() -> None:
    traj = almgren_chriss_trajectory(100.0, 10, urgency=0.0)
    # temporary = eta * sum(n_k^2) = eta * X^2 / N for equal slices
    assert almgren_chriss_cost(traj, eta=0.1, gamma=0.0) == pytest.approx(0.1 * 100**2 / 10)


def test_cost_includes_permanent_term() -> None:
    traj = almgren_chriss_trajectory(100.0, 10, urgency=0.0)
    temp = almgren_chriss_cost(traj, eta=0.1, gamma=0.0)
    total = almgren_chriss_cost(traj, eta=0.1, gamma=0.001)
    assert total == pytest.approx(temp + 0.5 * 0.001 * 100**2)


def test_front_loaded_costs_more_than_twap() -> None:
    twap = almgren_chriss_trajectory(100.0, 10, urgency=0.0)
    urgent = almgren_chriss_trajectory(100.0, 10, urgency=1.5)
    assert almgren_chriss_cost(urgent, eta=0.1) > almgren_chriss_cost(twap, eta=0.1)


def test_cost_trivial_trajectory_is_zero() -> None:
    assert almgren_chriss_cost(np.array([100.0]), eta=0.1) == 0.0
