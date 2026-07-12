"""Tests for the Reg-T margin account model."""

import pytest

from src.oms import MarginRequirements, Portfolio, Side, margin_report


def _portfolio() -> Portfolio:
    return Portfolio(initial_cash=100_000.0)


def test_flat_account_has_double_cash_buying_power() -> None:
    report = margin_report(_portfolio(), {})
    assert report.equity == 100_000.0
    assert report.initial_margin == 0.0
    assert report.excess_equity == 100_000.0
    assert report.buying_power == pytest.approx(200_000.0)  # Reg-T 2x
    assert not report.margin_call


def test_fully_invested_long_uses_half_the_equity() -> None:
    portfolio = _portfolio()
    portfolio.record_fill("AAA", Side.BUY, 1_000, 100.0)  # 100k long, cash 0
    report = margin_report(portfolio, {"AAA": 100.0})
    assert report.long_value == pytest.approx(100_000.0)
    assert report.initial_margin == pytest.approx(50_000.0)
    assert report.excess_equity == pytest.approx(50_000.0)
    assert report.buying_power == pytest.approx(100_000.0)
    assert not report.margin_call


def test_two_x_leverage_exhausts_buying_power() -> None:
    portfolio = _portfolio()
    portfolio.record_fill("AAA", Side.BUY, 2_000, 100.0)  # 200k long on 100k equity
    report = margin_report(portfolio, {"AAA": 100.0})
    assert report.excess_equity == pytest.approx(0.0)
    assert report.buying_power == 0.0
    assert not report.margin_call  # maintenance 50k < 100k equity


def test_over_levered_account_clamps_buying_power_at_zero() -> None:
    portfolio = _portfolio()
    portfolio.record_fill("AAA", Side.BUY, 3_000, 100.0)  # 300k long
    report = margin_report(portfolio, {"AAA": 100.0})
    assert report.excess_equity < 0
    assert report.buying_power == 0.0


def test_margin_call_when_equity_falls_below_maintenance() -> None:
    portfolio = _portfolio()
    portfolio.record_fill("AAA", Side.BUY, 4_000, 100.0)  # cash -300k
    at_cost = margin_report(portfolio, {"AAA": 100.0})
    # equity 100k == maintenance 100k -> not yet a call
    assert not at_cost.margin_call
    marked_down = margin_report(portfolio, {"AAA": 97.0})
    # equity 88k < maintenance 97k -> call
    assert marked_down.equity == pytest.approx(88_000.0)
    assert marked_down.maintenance_margin == pytest.approx(97_000.0)
    assert marked_down.margin_call


def test_short_side_uses_short_rates() -> None:
    portfolio = _portfolio()
    portfolio.record_fill("BBB", Side.SELL, 1_000, 100.0)  # 100k short, cash 200k
    report = margin_report(portfolio, {"BBB": 100.0})
    assert report.short_value == pytest.approx(100_000.0)
    assert report.initial_margin == pytest.approx(50_000.0)
    assert report.maintenance_margin == pytest.approx(30_000.0)
    assert not report.margin_call


def test_unmarked_positions_are_ignored() -> None:
    portfolio = _portfolio()
    portfolio.record_fill("AAA", Side.BUY, 100, 100.0)
    report = margin_report(portfolio, {})  # no marks at all
    assert report.long_value == 0.0
    assert report.initial_margin == 0.0


def test_custom_requirements_change_the_numbers() -> None:
    portfolio = _portfolio()
    portfolio.record_fill("AAA", Side.BUY, 1_000, 100.0)
    strict = MarginRequirements(initial_long=1.0, maintenance_long=0.5)
    report = margin_report(portfolio, {"AAA": 100.0}, requirements=strict)
    # cash account: full notional required up front -> no extra buying power
    assert report.initial_margin == pytest.approx(100_000.0)
    assert report.buying_power == 0.0


def test_invalid_requirements_raise() -> None:
    with pytest.raises(ValueError, match="initial_long"):
        MarginRequirements(initial_long=0.0)
    with pytest.raises(ValueError, match="maintenance_long"):
        MarginRequirements(maintenance_long=1.5)
    with pytest.raises(ValueError, match="cannot exceed"):
        MarginRequirements(initial_long=0.3, maintenance_long=0.4)
    with pytest.raises(ValueError, match="cannot exceed"):
        MarginRequirements(initial_short=0.3, maintenance_short=0.4)
