"""Tests for order netting."""

import pytest

from src.oms import Order, OrderType, Side, net_orders


def _mkt(symbol: str, side: Side, qty: float) -> Order:
    return Order(symbol=symbol, side=side, quantity=qty, order_type=OrderType.MARKET)


def test_offsetting_orders_cancel_out() -> None:
    batch = [_mkt("AAA", Side.BUY, 100), _mkt("AAA", Side.SELL, 100)]
    result = net_orders(batch)
    assert result.orders == []
    assert result.gross_quantity == 200.0
    assert result.net_quantity == 0.0
    assert result.reduction == pytest.approx(1.0)


def test_partial_offset_leaves_residual() -> None:
    batch = [_mkt("AAA", Side.BUY, 150), _mkt("AAA", Side.SELL, 40)]
    result = net_orders(batch)
    assert len(result.orders) == 1
    net = result.orders[0]
    assert net.symbol == "AAA"
    assert net.side is Side.BUY
    assert net.quantity == pytest.approx(110.0)
    assert net.client_tag == "netted"
    assert result.reduction == pytest.approx(1.0 - 110.0 / 190.0)


def test_multiple_symbols_netted_independently() -> None:
    batch = [
        _mkt("AAA", Side.BUY, 100),
        _mkt("BBB", Side.SELL, 50),
        _mkt("AAA", Side.BUY, 30),
        _mkt("BBB", Side.BUY, 10),
    ]
    result = net_orders(batch)
    by_symbol = {o.symbol: o for o in result.orders}
    assert by_symbol["AAA"].side is Side.BUY
    assert by_symbol["AAA"].quantity == pytest.approx(130.0)
    assert by_symbol["BBB"].side is Side.SELL
    assert by_symbol["BBB"].quantity == pytest.approx(40.0)


def test_orders_are_sorted_by_symbol() -> None:
    batch = [_mkt("ZZZ", Side.BUY, 5), _mkt("AAA", Side.BUY, 5), _mkt("MMM", Side.BUY, 5)]
    result = net_orders(batch)
    assert [o.symbol for o in result.orders] == ["AAA", "MMM", "ZZZ"]


def test_no_offsets_gives_zero_reduction() -> None:
    batch = [_mkt("AAA", Side.BUY, 100), _mkt("BBB", Side.BUY, 50)]
    result = net_orders(batch)
    assert result.reduction == 0.0
    assert result.gross_quantity == result.net_quantity == 150.0
    assert len(result.orders) == 2


def test_empty_batch() -> None:
    result = net_orders([])
    assert result.orders == []
    assert result.gross_quantity == 0.0
    assert result.reduction == 0.0


def test_side_flip_when_sells_dominate() -> None:
    batch = [_mkt("AAA", Side.BUY, 30), _mkt("AAA", Side.SELL, 100)]
    result = net_orders(batch)
    assert result.orders[0].side is Side.SELL
    assert result.orders[0].quantity == pytest.approx(70.0)


def test_inputs_are_not_mutated() -> None:
    batch = [_mkt("AAA", Side.BUY, 100), _mkt("AAA", Side.SELL, 40)]
    net_orders(batch)
    assert batch[0].quantity == 100.0
    assert batch[1].quantity == 40.0


def test_non_market_order_raises() -> None:
    limit = Order(
        symbol="AAA", side=Side.BUY, quantity=100, order_type=OrderType.LIMIT, limit_price=50.0
    )
    with pytest.raises(ValueError, match="MARKET orders"):
        net_orders([_mkt("AAA", Side.BUY, 100), limit])
