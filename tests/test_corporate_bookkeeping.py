"""Tests for corporate-action bookkeeping on the OMS portfolio."""

import pytest

from src.oms import Portfolio, Side, apply_dividend, apply_split


def _long_portfolio(qty: float = 100.0, price: float = 100.0) -> Portfolio:
    portfolio = Portfolio(initial_cash=100_000.0)
    portfolio.record_fill("AAA", Side.BUY, qty, price)
    return portfolio


def test_split_doubles_quantity_and_halves_avg_price() -> None:
    portfolio = _long_portfolio()
    new_qty = apply_split(portfolio, "AAA", ratio=2.0)
    position = portfolio.positions["AAA"]
    assert new_qty == pytest.approx(200.0)
    assert position.quantity == pytest.approx(200.0)
    assert position.avg_price == pytest.approx(50.0)
    assert position.cost_basis == pytest.approx(10_000.0)  # invariant


def test_split_preserves_equity_at_the_adjusted_mark() -> None:
    portfolio = _long_portfolio()
    equity_before = portfolio.equity({"AAA": 110.0})
    apply_split(portfolio, "AAA", ratio=2.0)
    equity_after = portfolio.equity({"AAA": 55.0})  # market halves the price
    assert equity_after == pytest.approx(equity_before)


def test_reverse_split_preserves_unrealized_pnl() -> None:
    portfolio = _long_portfolio()
    pnl_before = portfolio.positions["AAA"].unrealized_pnl(120.0)
    apply_split(portfolio, "AAA", ratio=0.25)  # 1-for-4
    pnl_after = portfolio.positions["AAA"].unrealized_pnl(480.0)
    assert pnl_after == pytest.approx(pnl_before)


def test_dividend_credits_longs() -> None:
    portfolio = _long_portfolio()
    cash_before = portfolio.cash
    flow = apply_dividend(portfolio, "AAA", amount_per_share=1.5)
    assert flow == pytest.approx(150.0)
    assert portfolio.cash == pytest.approx(cash_before + 150.0)
    assert portfolio.fees_paid == 0.0  # a dividend is not a fee


def test_dividend_debits_shorts() -> None:
    portfolio = Portfolio(initial_cash=100_000.0)
    portfolio.record_fill("BBB", Side.SELL, 200, 50.0)
    cash_before = portfolio.cash
    flow = apply_dividend(portfolio, "BBB", amount_per_share=1.0)
    assert flow == pytest.approx(-200.0)  # payment in lieu
    assert portfolio.cash == pytest.approx(cash_before - 200.0)


def test_events_on_unheld_symbols_are_no_ops() -> None:
    portfolio = _long_portfolio()
    cash_before = portfolio.cash
    assert apply_split(portfolio, "ZZZ", ratio=2.0) == 0.0
    assert apply_dividend(portfolio, "ZZZ", amount_per_share=5.0) == 0.0
    assert portfolio.cash == cash_before
    assert "ZZZ" not in portfolio.positions or portfolio.positions["ZZZ"].is_flat


def test_split_then_sell_realizes_correct_pnl() -> None:
    portfolio = _long_portfolio()  # 100 @ 100
    apply_split(portfolio, "AAA", ratio=2.0)  # 200 @ 50
    realized = portfolio.record_fill("AAA", Side.SELL, 200, 60.0)
    # bought for 10k, sold for 12k -> 2k profit regardless of the split
    assert realized == pytest.approx(2_000.0)


def test_bad_inputs_raise() -> None:
    portfolio = _long_portfolio()
    with pytest.raises(ValueError, match="ratio"):
        apply_split(portfolio, "AAA", ratio=0.0)
    with pytest.raises(ValueError, match="ratio"):
        apply_split(portfolio, "AAA", ratio=float("nan"))
    with pytest.raises(ValueError, match="amount_per_share"):
        apply_dividend(portfolio, "AAA", amount_per_share=-1.0)
