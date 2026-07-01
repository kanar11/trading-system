"""Tests for the Ralph Vince optimal-f sizer."""

import pytest

from src.risk.sizing import optimal_f


def test_known_optimum() -> None:
    # trades [+2, -1]: TWR(f) = (1 + 2f)(1 - f), maximised at f = 0.25
    assert optimal_f([2.0, -1.0]) == pytest.approx(0.25, abs=0.01)


def test_capped() -> None:
    assert optimal_f([2.0, -1.0], cap=0.1) == pytest.approx(0.1)


def test_no_losses_returns_cap() -> None:
    # unbounded growth -> saturates at the cap
    assert optimal_f([0.5, 1.0, 0.2], cap=1.0) == pytest.approx(1.0)


def test_no_edge_returns_zero() -> None:
    assert optimal_f([1.0, -1.0]) == 0.0  # zero mean
    assert optimal_f([-0.5, -0.2, -1.0]) == 0.0  # all losers


def test_empty_returns_zero() -> None:
    assert optimal_f([]) == 0.0


def test_result_in_unit_interval() -> None:
    f = optimal_f([3.0, -1.0, 2.0, -1.5, 1.0])
    assert 0.0 < f <= 1.0
