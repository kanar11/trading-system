"""Tests for the commission fee schedule."""

from datetime import datetime

import pytest

from src.oms import FeeSchedule, total_commission
from src.oms.order import Fill, Liquidity


def _fill(quantity: float, price: float, seq: int = 1) -> Fill:
    return Fill(
        seq=seq,
        ts=datetime(2024, 1, 2, 15, 30),
        quantity=quantity,
        price=price,
        liquidity=Liquidity.TAKER,
    )


def test_zero_schedule_is_free() -> None:
    schedule = FeeSchedule.zero()
    assert schedule.commission(1_000, 50.0) == 0.0


def test_per_share_with_minimum_floor() -> None:
    schedule = FeeSchedule.per_share_plan(rate=0.005, minimum=1.0)
    # 100 shares * 0.005 = 0.50 -> floored to the 1.00 minimum
    assert schedule.commission(100, 20.0) == 1.0
    # 1000 shares * 0.005 = 5.00 -> above the minimum
    assert schedule.commission(1_000, 20.0) == 5.0


def test_bps_plan_charges_fraction_of_notional() -> None:
    schedule = FeeSchedule.bps_plan(bps=10.0)  # 10 bps = 0.1%
    assert schedule.commission(100, 50.0) == pytest.approx(5.0)  # 5000 * 0.001


def test_maximum_caps_the_fee() -> None:
    schedule = FeeSchedule(pct_notional=0.01, maximum=25.0)
    # 1% of 10_000 = 100 -> capped at 25
    assert schedule.commission(100, 100.0) == 25.0


def test_components_are_additive() -> None:
    schedule = FeeSchedule(per_order=1.0, per_share=0.01, pct_notional=0.001)
    # 1.0 + 200*0.01 + 200*10*0.001 = 1.0 + 2.0 + 2.0
    assert schedule.commission(200, 10.0) == pytest.approx(5.0)


def test_zero_quantity_is_free_even_with_minimum() -> None:
    schedule = FeeSchedule(per_order=5.0, minimum=10.0)
    assert schedule.commission(0, 100.0) == 0.0


def test_negative_quantity_or_price_raises() -> None:
    schedule = FeeSchedule.zero()
    with pytest.raises(ValueError, match="quantity"):
        schedule.commission(-1, 10.0)
    with pytest.raises(ValueError, match="price"):
        schedule.commission(1, -10.0)


def test_negative_component_raises() -> None:
    with pytest.raises(ValueError, match="per_share"):
        FeeSchedule(per_share=-0.01)


def test_maximum_below_minimum_raises() -> None:
    with pytest.raises(ValueError, match="maximum"):
        FeeSchedule(minimum=5.0, maximum=1.0)


def test_fill_commission_uses_fill_fields() -> None:
    schedule = FeeSchedule.per_share_plan(rate=0.005, minimum=1.0)
    assert schedule.fill_commission(_fill(1_000, 20.0)) == 5.0


def test_total_commission_sums_per_fill() -> None:
    schedule = FeeSchedule.per_share_plan(rate=0.005, minimum=1.0)
    fills = [_fill(100, 20.0, seq=1), _fill(1_000, 20.0, seq=2)]
    # minimum applies per fill: 1.0 + 5.0
    assert total_commission(fills, schedule) == 6.0


def test_total_commission_empty_is_zero() -> None:
    assert total_commission([], FeeSchedule.per_share_plan()) == 0.0
