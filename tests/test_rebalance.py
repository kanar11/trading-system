"""Tests for the rebalance order generator."""

import math

import pytest

from src.oms import Portfolio, RebalanceOrder, Side, rebalance_orders


def _portfolio() -> Portfolio:
    return Portfolio(initial_cash=100_000.0)


def test_buy_into_target_from_all_cash() -> None:
    orders = rebalance_orders(_portfolio(), {"AAA": 0.5}, {"AAA": 100.0})
    assert orders == [
        RebalanceOrder(symbol="AAA", side=Side.BUY, quantity=500.0, notional=50_000.0)
    ]


def test_symbol_missing_from_targets_is_closed() -> None:
    portfolio = _portfolio()
    portfolio.record_fill("BBB", Side.BUY, 100, 50.0)
    orders = rebalance_orders(portfolio, {}, {"BBB": 50.0})
    assert len(orders) == 1
    assert orders[0].side is Side.SELL
    assert orders[0].quantity == pytest.approx(100.0)


def test_tops_up_to_target_weight() -> None:
    portfolio = _portfolio()
    portfolio.record_fill("AAA", Side.BUY, 100, 100.0)  # 10k position, 90k cash
    orders = rebalance_orders(portfolio, {"AAA": 0.2}, {"AAA": 100.0})
    # equity 100k -> target 200 shares, held 100 -> buy 100 more
    assert len(orders) == 1
    assert orders[0].side is Side.BUY
    assert orders[0].quantity == pytest.approx(100.0)


def test_negative_weight_opens_a_short() -> None:
    orders = rebalance_orders(_portfolio(), {"AAA": -0.1}, {"AAA": 100.0})
    assert len(orders) == 1
    assert orders[0].side is Side.SELL
    assert orders[0].quantity == pytest.approx(100.0)


def test_already_at_target_emits_nothing() -> None:
    portfolio = _portfolio()
    portfolio.record_fill("AAA", Side.BUY, 100, 100.0)
    orders = rebalance_orders(portfolio, {"AAA": 0.1}, {"AAA": 100.0})
    assert orders == []


def test_min_notional_skips_dust() -> None:
    portfolio = _portfolio()
    portfolio.record_fill("AAA", Side.BUY, 100, 100.0)
    # target 0.101 -> delta 1 share = 100 notional, below the 500 floor
    orders = rebalance_orders(portfolio, {"AAA": 0.101}, {"AAA": 100.0}, min_notional=500.0)
    assert orders == []


def test_lot_size_floors_the_quantity() -> None:
    orders = rebalance_orders(_portfolio(), {"AAA": 0.10557}, {"AAA": 100.0}, lot_size=1.0)
    assert len(orders) == 1
    assert orders[0].quantity == pytest.approx(105.0)
    assert math.isclose(orders[0].notional, 10_500.0)


def test_lot_rounding_to_zero_is_skipped() -> None:
    # delta of 0.5 shares floors to zero whole lots
    orders = rebalance_orders(_portfolio(), {"AAA": 0.0005}, {"AAA": 100.0}, lot_size=1.0)
    assert orders == []


def test_orders_sorted_by_symbol() -> None:
    orders = rebalance_orders(
        _portfolio(), {"ZZZ": 0.1, "AAA": 0.1, "MMM": 0.1}, {"AAA": 10.0, "MMM": 10.0, "ZZZ": 10.0}
    )
    assert [o.symbol for o in orders] == ["AAA", "MMM", "ZZZ"]


def test_missing_or_bad_mark_raises() -> None:
    with pytest.raises(ValueError, match="missing mark"):
        rebalance_orders(_portfolio(), {"AAA": 0.5}, {})
    with pytest.raises(ValueError, match="> 0"):
        rebalance_orders(_portfolio(), {"AAA": 0.5}, {"AAA": 0.0})


def test_non_finite_weight_raises() -> None:
    with pytest.raises(ValueError, match="finite"):
        rebalance_orders(_portfolio(), {"AAA": float("nan")}, {"AAA": 100.0})


def test_negative_tolerances_raise() -> None:
    with pytest.raises(ValueError, match="min_notional"):
        rebalance_orders(_portfolio(), {}, {}, min_notional=-1.0)
    with pytest.raises(ValueError, match="lot_size"):
        rebalance_orders(_portfolio(), {}, {}, lot_size=-1.0)


def test_non_positive_equity_raises() -> None:
    portfolio = _portfolio()
    portfolio.record_fill("AAA", Side.SELL, 1_000, 100.0)  # short 1000 @ 100
    # marked at 250 the short is deeply under water: equity 200k - 250k < 0
    with pytest.raises(ValueError, match="equity"):
        rebalance_orders(portfolio, {"AAA": -1.0}, {"AAA": 250.0})
