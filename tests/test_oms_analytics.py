"""Tests for portfolio exposure / concentration analytics."""

import pytest

from src.oms import Portfolio, Side, portfolio_exposure


def test_long_short_exposures_exact() -> None:
    p = Portfolio(initial_cash=100_000)
    p.record_fill("SPY", Side.BUY, 100, 400.0)
    p.record_fill("QQQ", Side.SELL, 50, 300.0)
    rep = portfolio_exposure(p, {"SPY": 410.0, "QQQ": 310.0})

    assert rep.gross_exposure == pytest.approx(56_500)  # 41000 + 15500
    assert rep.net_exposure == pytest.approx(25_500)  # 41000 - 15500
    assert rep.long_exposure == pytest.approx(41_000)
    assert rep.short_exposure == pytest.approx(15_500)
    assert rep.n_long == 1
    assert rep.n_short == 1
    # cash: 100000 - 40000 (buy) + 15000 (sell) = 75000; equity = 75000 + 25500
    assert rep.leverage == pytest.approx(56_500 / 100_500)
    w_long, w_short = 41_000 / 56_500, 15_500 / 56_500
    assert rep.concentration_hhi == pytest.approx(w_long**2 + w_short**2)
    assert rep.largest_weight == pytest.approx(w_long)


def test_empty_portfolio_is_all_zero() -> None:
    rep = portfolio_exposure(Portfolio(initial_cash=100_000), {})
    assert rep.gross_exposure == 0.0
    assert rep.net_exposure == 0.0
    assert rep.leverage == 0.0
    assert rep.concentration_hhi == 0.0
    assert rep.largest_weight == 0.0
    assert rep.n_long == 0
    assert rep.n_short == 0


def test_single_position_is_fully_concentrated() -> None:
    p = Portfolio()
    p.record_fill("SPY", Side.BUY, 10, 100.0)
    rep = portfolio_exposure(p, {"SPY": 100.0})
    assert rep.concentration_hhi == pytest.approx(1.0)
    assert rep.largest_weight == pytest.approx(1.0)
    assert rep.short_exposure == 0.0


def test_flat_positions_excluded() -> None:
    p = Portfolio()
    p.record_fill("SPY", Side.BUY, 10, 100.0)
    p.record_fill("SPY", Side.SELL, 10, 110.0)  # closed -> flat
    rep = portfolio_exposure(p, {"SPY": 110.0})
    assert rep.gross_exposure == 0.0
    assert rep.n_long == 0


def test_unmarked_symbols_excluded() -> None:
    p = Portfolio()
    p.record_fill("SPY", Side.BUY, 10, 100.0)
    p.record_fill("QQQ", Side.BUY, 5, 200.0)
    rep = portfolio_exposure(p, {"SPY": 100.0})  # QQQ has no mark
    assert rep.n_long == 1
    assert rep.gross_exposure == pytest.approx(1_000.0)


def test_long_only_net_equals_gross() -> None:
    p = Portfolio()
    p.record_fill("A", Side.BUY, 10, 100.0)
    p.record_fill("B", Side.BUY, 5, 100.0)
    rep = portfolio_exposure(p, {"A": 100.0, "B": 100.0})
    assert rep.short_exposure == 0.0
    assert rep.n_long == 2
    assert rep.net_exposure == pytest.approx(rep.gross_exposure)
