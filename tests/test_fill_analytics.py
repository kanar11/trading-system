"""Tests for OMS fill analytics."""

from datetime import datetime, timedelta

import pytest

from src.oms import Fill, Liquidity, Order, Side, summarize_fills

_T0 = datetime(2020, 1, 1, 9, 30)


def test_summary_values() -> None:
    fills = [
        Fill(seq=1, ts=_T0, quantity=10, price=100.0, liquidity=Liquidity.TAKER),
        Fill(
            seq=2,
            ts=_T0 + timedelta(seconds=60),
            quantity=30,
            price=102.0,
            liquidity=Liquidity.MAKER,
        ),
    ]
    s = summarize_fills(fills)
    assert s.n_fills == 2
    assert s.total_quantity == pytest.approx(40.0)
    assert s.vwap == pytest.approx((1000.0 + 3060.0) / 40.0)  # 101.5
    assert s.maker_fraction == pytest.approx(0.75)
    assert s.duration_seconds == pytest.approx(60.0)


def test_empty_is_zero() -> None:
    s = summarize_fills([])
    assert s.n_fills == 0
    assert s.total_quantity == 0.0
    assert s.vwap == 0.0
    assert s.maker_fraction == 0.0
    assert s.duration_seconds == 0.0


def test_all_taker_single_fill() -> None:
    s = summarize_fills([Fill(seq=1, ts=_T0, quantity=5, price=100.0, liquidity=Liquidity.TAKER)])
    assert s.maker_fraction == 0.0
    assert s.duration_seconds == 0.0
    assert s.vwap == pytest.approx(100.0)


def test_all_maker() -> None:
    fills = [
        Fill(seq=1, ts=_T0, quantity=5, price=100.0, liquidity=Liquidity.MAKER),
        Fill(seq=2, ts=_T0, quantity=5, price=101.0, liquidity=Liquidity.MAKER),
    ]
    assert summarize_fills(fills).maker_fraction == pytest.approx(1.0)


def test_summarizes_order_fills() -> None:
    o = Order(symbol="SPY", side=Side.BUY, quantity=40)
    o.record_fill(10, 100.0, when=_T0, liquidity=Liquidity.TAKER)
    o.record_fill(30, 102.0, when=_T0 + timedelta(seconds=60), liquidity=Liquidity.MAKER)
    s = summarize_fills(o.fills)
    assert s.n_fills == 2
    assert s.vwap == pytest.approx(101.5)
    assert s.maker_fraction == pytest.approx(0.75)
    assert s.duration_seconds == pytest.approx(60.0)
