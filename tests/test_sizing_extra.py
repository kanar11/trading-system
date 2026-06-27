"""Tests for the additional position sizers (vol target, CPPI, dd throttle)."""

import pytest

from src.risk.sizing import cppi_fraction, drawdown_throttle, volatility_target_size

# --- volatility_target_size ------------------------------------------------


def test_vol_target_matches_when_equal() -> None:
    assert volatility_target_size(0.15, 0.15) == pytest.approx(1.0)


def test_vol_target_halves_for_double_vol() -> None:
    assert volatility_target_size(0.30, 0.15) == pytest.approx(0.5)


def test_vol_target_clamped_by_max_size() -> None:
    assert volatility_target_size(0.075, 0.15, max_size=1.0) == pytest.approx(1.0)
    assert volatility_target_size(0.075, 0.15, max_size=2.0) == pytest.approx(2.0)


def test_vol_target_guards() -> None:
    assert volatility_target_size(0.0, 0.15) == 0.0
    assert volatility_target_size(0.15, 0.0) == 0.0


# --- cppi_fraction ---------------------------------------------------------


def test_cppi_cushion_exposure() -> None:
    assert cppi_fraction(100.0, 80.0, multiplier=3.0) == pytest.approx(0.6)


def test_cppi_zero_at_or_below_floor() -> None:
    assert cppi_fraction(80.0, 80.0) == 0.0
    assert cppi_fraction(70.0, 80.0) == 0.0


def test_cppi_clamped_by_max_size() -> None:
    assert cppi_fraction(100.0, 80.0, multiplier=10.0, max_size=1.0) == pytest.approx(1.0)


def test_cppi_guards() -> None:
    assert cppi_fraction(0.0, -10.0) == 0.0
    assert cppi_fraction(100.0, 80.0, multiplier=0.0) == 0.0


# --- drawdown_throttle -----------------------------------------------------


def test_throttle_full_at_zero_drawdown() -> None:
    assert drawdown_throttle(0.0, 0.2) == pytest.approx(1.0)


def test_throttle_zero_at_max_drawdown() -> None:
    assert drawdown_throttle(0.2, 0.2) == pytest.approx(0.0)
    assert drawdown_throttle(-0.2, 0.2) == pytest.approx(0.0)  # sign-insensitive


def test_throttle_linear_halfway() -> None:
    assert drawdown_throttle(0.1, 0.2) == pytest.approx(0.5)


def test_throttle_beyond_max_is_zero() -> None:
    assert drawdown_throttle(0.3, 0.2) == 0.0


def test_throttle_scales_with_max_size() -> None:
    assert drawdown_throttle(0.1, 0.2, max_size=2.0) == pytest.approx(1.0)


def test_throttle_guard_on_bad_max() -> None:
    assert drawdown_throttle(0.05, 0.0) == 0.0
