"""HFT-grade behaviour tests for the hardened Order state machine.

These complement the lifecycle tests in ``test_oms.py`` and lock down
the production-grade guarantees: strict transitions, sequenced/immutable
fill audit trail, monotonic timestamps, structured rejects, amend
(cancel-replace), float-drift-safe averaging and FIX-aligned accessors.
"""

import dataclasses
from datetime import datetime, timedelta

import pytest

from src.oms import (
    Fill,
    IllegalOrderTransition,
    Liquidity,
    Order,
    OrderStatus,
    OrderType,
    RejectReason,
    Side,
)


def _mkt(qty=100):
    return Order(symbol="SPY", side=Side.BUY, quantity=qty)


# ---------------------------------------------------------------------------
# strict state machine
# ---------------------------------------------------------------------------


class TestStateMachine:
    def test_fill_after_cancel_raises(self):
        o = _mkt()
        o.cancel()
        with pytest.raises(IllegalOrderTransition):
            o.record_fill(10, 100.0)

    def test_fill_after_full_fill_raises(self):
        o = _mkt(10)
        o.record_fill(10, 100.0)
        assert o.status is OrderStatus.FILLED
        with pytest.raises(IllegalOrderTransition):
            o.record_fill(1, 100.0)

    def test_activate_moves_pending_to_working(self):
        o = _mkt()
        assert o.status is OrderStatus.PENDING
        o.activate()
        assert o.status is OrderStatus.WORKING

    def test_activate_is_noop_when_not_pending(self):
        o = _mkt(10)
        o.record_fill(5, 100.0)  # now PARTIALLY_FILLED
        o.activate()
        assert o.status is OrderStatus.PARTIALLY_FILLED

    def test_partial_then_full_progression(self):
        o = _mkt(100)
        o.record_fill(40, 100.0)
        assert o.status is OrderStatus.PARTIALLY_FILLED
        o.record_fill(60, 100.0)
        assert o.status is OrderStatus.FILLED


# ---------------------------------------------------------------------------
# sequenced, immutable, tuple-compatible fills
# ---------------------------------------------------------------------------


class TestFillAuditTrail:
    def test_fills_are_sequenced(self):
        o = _mkt(100)
        o.record_fill(30, 100.0)
        o.record_fill(30, 101.0)
        o.record_fill(40, 102.0)
        assert [f.seq for f in o.fills] == [1, 2, 3]

    def test_fill_is_immutable(self):
        o = _mkt(10)
        o.record_fill(10, 100.0)
        f = o.fills[0]
        with pytest.raises(dataclasses.FrozenInstanceError):
            f.price = 999.0

    def test_fill_tuple_unpacking_backcompat(self):
        # old call sites do: for ts, qty, price in order.fills
        o = _mkt(10)
        when = datetime(2024, 1, 1)
        o.record_fill(10, 100.0, when=when)
        ts, qty, price = o.fills[0]
        assert ts == when and qty == 10 and price == 100.0

    def test_record_fill_returns_fill(self):
        o = _mkt(10)
        f = o.record_fill(10, 100.0)
        assert isinstance(f, Fill)
        assert f.quantity == 10 and f.price == 100.0

    def test_liquidity_flag_recorded(self):
        o = _mkt(10)
        o.record_fill(10, 100.0, liquidity=Liquidity.MAKER)
        assert o.fills[0].liquidity is Liquidity.MAKER

    def test_liquidity_accepts_string(self):
        o = _mkt(10)
        o.record_fill(10, 100.0, liquidity="MAKER")
        assert o.fills[0].liquidity is Liquidity.MAKER


# ---------------------------------------------------------------------------
# monotonic timestamps
# ---------------------------------------------------------------------------


class TestMonotonicTimestamps:
    def test_out_of_order_fill_rejected(self):
        o = _mkt(100)
        t0 = datetime(2024, 1, 2)
        o.record_fill(50, 100.0, when=t0)
        with pytest.raises(ValueError, match="out-of-order"):
            o.record_fill(50, 100.0, when=t0 - timedelta(days=1))

    def test_equal_timestamps_allowed(self):
        o = _mkt(100)
        t0 = datetime(2024, 1, 2)
        o.record_fill(50, 100.0, when=t0)
        o.record_fill(50, 100.0, when=t0)  # same ts is fine
        assert o.status is OrderStatus.FILLED


# ---------------------------------------------------------------------------
# float-drift-safe average price
# ---------------------------------------------------------------------------


class TestAveragePrice:
    def test_avg_price_weighted(self):
        o = _mkt(100)
        o.record_fill(50, 100.0)
        o.record_fill(50, 200.0)
        assert o.avg_fill_price == pytest.approx(150.0)

    def test_avg_price_no_drift_over_many_fills(self):
        # 1000 fills of 0.001 each at price 3.33 — average must stay exact
        o = Order(symbol="X", side=Side.BUY, quantity=1.0)
        for _ in range(1000):
            o.record_fill(0.001, 3.33)
        assert o.avg_fill_price == pytest.approx(3.33, abs=1e-9)
        assert o.is_complete


# ---------------------------------------------------------------------------
# FIX-aligned accessors
# ---------------------------------------------------------------------------


class TestFixAliases:
    def test_cum_and_leaves_qty(self):
        o = _mkt(100)
        o.record_fill(30, 100.0)
        assert o.cum_qty == 30
        assert o.leaves_qty == 70
        assert o.avg_px == pytest.approx(100.0)

    def test_side_helpers(self):
        assert _mkt().is_buy
        assert Order(symbol="X", side=Side.SELL, quantity=1).is_sell
        assert Side.BUY.opposite is Side.SELL


# ---------------------------------------------------------------------------
# structured rejects
# ---------------------------------------------------------------------------


class TestReject:
    def test_reject_with_enum_reason(self):
        o = _mkt()
        assert o.reject(RejectReason.RISK_LIMIT, "max position exceeded") is True
        assert o.status is OrderStatus.REJECTED
        assert o.reject_reason is RejectReason.RISK_LIMIT
        assert "max position" in o.reject_detail

    def test_reject_with_string_maps_to_unknown(self):
        o = _mkt()
        o.reject("some freeform message")
        assert o.reject_reason is RejectReason.UNKNOWN
        assert o.reject_detail == "some freeform message"

    def test_reject_does_not_corrupt_client_tag(self):
        o = Order(symbol="SPY", side=Side.BUY, quantity=10, client_tag="strat-A")
        o.reject(RejectReason.INVALID_PRICE)
        assert o.client_tag == "strat-A"  # untouched

    def test_reject_terminal_is_noop(self):
        o = _mkt(10)
        o.record_fill(10, 100.0)
        assert o.reject(RejectReason.RISK_LIMIT) is False
        assert o.status is OrderStatus.FILLED


# ---------------------------------------------------------------------------
# amend (cancel-replace)
# ---------------------------------------------------------------------------


class TestAmend:
    def test_amend_quantity_bumps_version(self):
        o = _mkt(100)
        assert o.version == 0
        v = o.amend(new_quantity=150)
        assert v == 1
        assert o.quantity == 150

    def test_amend_below_filled_rejected(self):
        o = _mkt(100)
        o.record_fill(60, 100.0)
        with pytest.raises(ValueError, match="below already-filled"):
            o.amend(new_quantity=50)

    def test_amend_down_to_filled_completes_order(self):
        o = _mkt(100)
        o.record_fill(60, 100.0)
        o.amend(new_quantity=60)
        assert o.status is OrderStatus.FILLED

    def test_amend_limit_price(self):
        o = Order(
            symbol="SPY", side=Side.BUY, quantity=10, order_type=OrderType.LIMIT, limit_price=400.0
        )
        o.amend(new_limit_price=395.0)
        assert o.limit_price == 395.0

    def test_amend_limit_on_market_order_rejected(self):
        o = _mkt()
        with pytest.raises(ValueError, match="cannot set limit_price"):
            o.amend(new_limit_price=395.0)

    def test_amend_terminal_raises(self):
        o = _mkt(10)
        o.record_fill(10, 100.0)
        with pytest.raises(IllegalOrderTransition):
            o.amend(new_quantity=20)


# ---------------------------------------------------------------------------
# serialisation
# ---------------------------------------------------------------------------


class TestSerialization:
    def test_to_dict_has_fix_fields(self):
        o = _mkt(100)
        o.record_fill(40, 100.0)
        d = o.to_dict()
        assert d["cum_qty"] == 40
        assert d["leaves_qty"] == 60
        assert d["status"] == "PARTIALLY_FILLED"
        assert d["n_fills"] == 1
        assert d["reject_reason"] == ""

    def test_slots_prevents_typo_attributes(self):
        # slots=True means stray attribute assignment fails loudly —
        # an important guard against "I set the wrong field" bugs.
        o = _mkt()
        with pytest.raises(AttributeError):
            o.filed_quantity = 5  # typo of filled_quantity
