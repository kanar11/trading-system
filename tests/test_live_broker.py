"""Tests for the paper-broker live adapter stub."""

import pytest

from src.live.broker import Broker, PaperBroker, BrokerFill
from src.oms import Order, OrderStatus, OrderType, Side, TimeInForce


class TestPaperBrokerBasics:
    def test_starts_with_initial_cash(self):
        b = PaperBroker(initial_cash=50_000)
        assert b.cash() == 50_000
        assert b.equity({}) == 50_000

    def test_submit_market_fills_at_mark(self):
        b = PaperBroker(initial_cash=100_000)
        o = Order(symbol="SPY", side=Side.BUY, quantity=10)
        b.submit_order(o, mark_price=400.0)
        assert o.status is OrderStatus.FILLED
        assert b.cash() == pytest.approx(100_000 - 4000)
        assert b.positions()["SPY"] == 10

    def test_submit_without_mark_leaves_working(self):
        b = PaperBroker()
        o = Order(symbol="SPY", side=Side.BUY, quantity=10)
        b.submit_order(o)
        assert o.status is OrderStatus.WORKING
        assert b.open_orders() == [o]

    def test_commission_charged(self):
        b = PaperBroker(initial_cash=100_000, commission_per_share=0.01)
        o = Order(symbol="SPY", side=Side.BUY, quantity=100)
        b.submit_order(o, mark_price=400.0)
        # commission 100 * 0.01 = 1
        assert b.cash() == pytest.approx(100_000 - 40_000 - 1)


class TestLimitOrders:
    def test_limit_does_not_fill_above_for_buy(self):
        b = PaperBroker(initial_cash=100_000)
        o = Order(symbol="SPY", side=Side.BUY, quantity=10,
                  order_type=OrderType.LIMIT, limit_price=400.0)
        b.submit_order(o, mark_price=410.0)
        assert o.status is OrderStatus.WORKING
        # mark drops to 400 → fills
        b.poll({"SPY": 400.0})
        assert o.status is OrderStatus.FILLED

    def test_limit_does_not_fill_below_for_sell(self):
        b = PaperBroker(initial_cash=100_000)
        # we need a long position to sell from — open one first
        buy = Order(symbol="SPY", side=Side.BUY, quantity=10)
        b.submit_order(buy, mark_price=400.0)
        sell = Order(symbol="SPY", side=Side.SELL, quantity=10,
                     order_type=OrderType.LIMIT, limit_price=420.0)
        b.submit_order(sell, mark_price=410.0)
        assert sell.status is OrderStatus.WORKING
        b.poll({"SPY": 425.0})
        assert sell.status is OrderStatus.FILLED


class TestStopOrders:
    def test_stop_buy_triggers_above(self):
        b = PaperBroker(initial_cash=100_000)
        o = Order(symbol="SPY", side=Side.BUY, quantity=10,
                  order_type=OrderType.STOP, stop_price=410.0)
        b.submit_order(o, mark_price=405.0)
        assert o.status is OrderStatus.WORKING
        b.poll({"SPY": 415.0})  # above stop
        assert o.status is OrderStatus.FILLED


class TestCancellation:
    def test_cancel_working_order(self):
        b = PaperBroker()
        o = Order(symbol="SPY", side=Side.BUY, quantity=10,
                  order_type=OrderType.LIMIT, limit_price=100.0)
        b.submit_order(o, mark_price=110.0)  # too high → working
        assert b.cancel_order(o.order_id) is True
        assert o.status is OrderStatus.CANCELLED

    def test_cancel_already_filled_is_noop(self):
        b = PaperBroker(initial_cash=100_000)
        o = Order(symbol="SPY", side=Side.BUY, quantity=10)
        b.submit_order(o, mark_price=400)
        assert b.cancel_order(o.order_id) is False


class TestBrokerInterface:
    def test_paper_broker_is_a_broker(self):
        assert isinstance(PaperBroker(), Broker)

    def test_open_orders_filtered_by_symbol(self):
        b = PaperBroker()
        a = Order(symbol="AAA", side=Side.BUY, quantity=10,
                  order_type=OrderType.LIMIT, limit_price=50)
        bb = Order(symbol="BBB", side=Side.BUY, quantity=10,
                   order_type=OrderType.LIMIT, limit_price=50)
        b.submit_order(a, mark_price=100)
        b.submit_order(bb, mark_price=100)
        assert len(b.open_orders()) == 2
        assert len(b.open_orders(symbol="AAA")) == 1

    def test_fills_recorded(self):
        b = PaperBroker(initial_cash=100_000)
        b.submit_order(Order(symbol="SPY", side=Side.BUY, quantity=5), mark_price=400)
        assert len(b.fills) == 1
        f = b.fills[0]
        assert isinstance(f, BrokerFill)
        assert f.quantity == 5 and f.price == 400
