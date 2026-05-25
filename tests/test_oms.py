"""Tests for the order-management primitives (Order, Position, Portfolio)."""

from datetime import datetime

import pytest

from src.oms import (
    Order, OrderStatus, OrderType, Side, TimeInForce,
    Position, Portfolio,
)


# ---------------------------------------------------------------------------
# Order
# ---------------------------------------------------------------------------

class TestOrder:
    def test_construct_market_order(self):
        o = Order(symbol="SPY", side=Side.BUY, quantity=100)
        assert o.order_type is OrderType.MARKET
        assert o.status is OrderStatus.PENDING
        assert o.remaining_quantity == 100

    def test_limit_requires_price(self):
        with pytest.raises(ValueError, match="limit_price"):
            Order(symbol="SPY", side=Side.BUY, quantity=10, order_type=OrderType.LIMIT)

    def test_stop_requires_stop_price(self):
        with pytest.raises(ValueError, match="stop_price"):
            Order(symbol="SPY", side=Side.SELL, quantity=10, order_type=OrderType.STOP)

    def test_negative_quantity_rejected(self):
        with pytest.raises(ValueError, match="quantity"):
            Order(symbol="SPY", side=Side.BUY, quantity=0)

    def test_signed_quantity(self):
        buy = Order(symbol="SPY", side=Side.BUY, quantity=10)
        sell = Order(symbol="SPY", side=Side.SELL, quantity=10)
        assert buy.signed_quantity == 10
        assert sell.signed_quantity == -10

    def test_partial_fill_transitions_status(self):
        o = Order(symbol="SPY", side=Side.BUY, quantity=100)
        o.record_fill(40, 400.0)
        assert o.status is OrderStatus.PARTIALLY_FILLED
        assert o.filled_quantity == 40
        assert o.remaining_quantity == 60
        assert o.avg_fill_price == 400.0

    def test_multi_fill_avg_price(self):
        o = Order(symbol="SPY", side=Side.BUY, quantity=100)
        o.record_fill(50, 100.0)
        o.record_fill(50, 200.0)
        assert o.status is OrderStatus.FILLED
        assert o.avg_fill_price == pytest.approx(150.0)

    def test_overfill_rejected(self):
        o = Order(symbol="SPY", side=Side.BUY, quantity=10)
        o.record_fill(10, 100.0)
        with pytest.raises(ValueError, match="over-fill"):
            o.record_fill(1, 100.0)

    def test_cancel(self):
        o = Order(symbol="SPY", side=Side.BUY, quantity=10)
        o.cancel()
        assert o.status is OrderStatus.CANCELLED
        assert o.status.is_terminal

    def test_cancel_after_fill_is_noop(self):
        o = Order(symbol="SPY", side=Side.BUY, quantity=10)
        o.record_fill(10, 100.0)
        o.cancel()
        assert o.status is OrderStatus.FILLED


# ---------------------------------------------------------------------------
# Position
# ---------------------------------------------------------------------------

class TestPosition:
    def test_opens_long(self):
        p = Position(symbol="SPY")
        realised = p.apply_fill(Side.BUY, 100, 400.0)
        assert p.quantity == 100
        assert p.avg_price == 400.0
        assert realised == 0.0

    def test_adds_to_long_recomputes_avg(self):
        p = Position(symbol="SPY")
        p.apply_fill(Side.BUY, 100, 400.0)
        p.apply_fill(Side.BUY, 100, 500.0)
        assert p.quantity == 200
        assert p.avg_price == pytest.approx(450.0)
        assert p.realized_pnl == 0.0

    def test_partial_close_realises_pnl(self):
        p = Position(symbol="SPY")
        p.apply_fill(Side.BUY, 100, 400.0)
        realised = p.apply_fill(Side.SELL, 40, 420.0)
        assert realised == pytest.approx(40 * 20)  # 40 shares * $20 profit
        assert p.quantity == 60
        assert p.avg_price == pytest.approx(400.0)  # unchanged on partial close

    def test_full_close_resets_avg(self):
        p = Position(symbol="SPY")
        p.apply_fill(Side.BUY, 100, 400.0)
        p.apply_fill(Side.SELL, 100, 420.0)
        assert p.is_flat
        assert p.avg_price == 0.0
        assert p.realized_pnl == pytest.approx(2000)

    def test_flip_long_to_short(self):
        p = Position(symbol="SPY")
        p.apply_fill(Side.BUY, 100, 400.0)
        realised = p.apply_fill(Side.SELL, 150, 420.0)
        # closed 100 at +20 each = 2000; opened 50 short at 420
        assert realised == pytest.approx(2000)
        assert p.quantity == pytest.approx(-50)
        assert p.avg_price == pytest.approx(420.0)

    def test_short_realises_on_buy_back(self):
        p = Position(symbol="SPY")
        p.apply_fill(Side.SELL, 100, 500.0)  # open short
        realised = p.apply_fill(Side.BUY, 100, 480.0)  # cover
        assert realised == pytest.approx(2000)  # short profited 20/share
        assert p.is_flat

    def test_unrealized_pnl_long(self):
        p = Position(symbol="SPY")
        p.apply_fill(Side.BUY, 100, 400.0)
        assert p.unrealized_pnl(450.0) == pytest.approx(5000)

    def test_unrealized_pnl_short(self):
        p = Position(symbol="SPY")
        p.apply_fill(Side.SELL, 100, 500.0)
        # price falls to 480 — short PnL = (480-500) * -100 = +2000
        assert p.unrealized_pnl(480.0) == pytest.approx(2000)

    def test_total_traded_quantity(self):
        p = Position(symbol="SPY")
        p.apply_fill(Side.BUY, 30, 400.0)
        p.apply_fill(Side.BUY, 70, 410.0)
        p.apply_fill(Side.SELL, 50, 420.0)
        assert p.total_traded_qty == 150


# ---------------------------------------------------------------------------
# Portfolio
# ---------------------------------------------------------------------------

class TestPortfolio:
    def test_starts_with_initial_cash(self):
        p = Portfolio(initial_cash=50_000)
        assert p.cash == 50_000
        assert p.equity({}) == 50_000

    def test_record_buy_decreases_cash_and_opens_position(self):
        p = Portfolio(initial_cash=100_000)
        p.record_fill("SPY", Side.BUY, 100, 400.0, commission=1.0)
        assert p.cash == pytest.approx(100_000 - 100 * 400 - 1)
        assert p.fees_paid == 1.0
        pos = p.positions["SPY"]
        assert pos.quantity == 100

    def test_record_sell_against_long(self):
        p = Portfolio(initial_cash=100_000)
        p.record_fill("SPY", Side.BUY, 100, 400.0)
        p.record_fill("SPY", Side.SELL, 50, 420.0, commission=0.5)
        pos = p.positions["SPY"]
        assert pos.quantity == 50
        assert pos.realized_pnl == pytest.approx(50 * 20)
        # cash: -40000 +21000 -0.5 = ?
        assert p.cash == pytest.approx(100_000 - 40_000 + 21_000 - 0.5)

    def test_equity_includes_unrealized(self):
        p = Portfolio(initial_cash=100_000)
        p.record_fill("SPY", Side.BUY, 100, 400.0)
        # equity = cash (60_000) + 100 * 450 = 105_000
        eq = p.equity({"SPY": 450.0})
        assert eq == pytest.approx(105_000)

    def test_mark_to_market_appends_history(self):
        p = Portfolio(initial_cash=100_000)
        p.record_fill("SPY", Side.BUY, 10, 400.0)
        t1 = datetime(2024, 1, 1)
        t2 = datetime(2024, 1, 2)
        eq1 = p.mark_to_market(t1, {"SPY": 400.0})
        eq2 = p.mark_to_market(t2, {"SPY": 410.0})
        assert eq1 == pytest.approx(100_000)
        assert eq2 == pytest.approx(100_100)
        assert len(p.equity_history) == 2

    def test_gross_vs_net_exposure_with_short(self):
        p = Portfolio(initial_cash=100_000)
        p.record_fill("AAA", Side.BUY, 10, 100.0)
        p.record_fill("BBB", Side.SELL, 10, 100.0)
        marks = {"AAA": 100.0, "BBB": 100.0}
        assert p.gross_exposure(marks) == pytest.approx(2000)
        assert p.net_exposure(marks) == pytest.approx(0)

    def test_rejects_zero_initial_cash(self):
        with pytest.raises(ValueError):
            Portfolio(initial_cash=0)
