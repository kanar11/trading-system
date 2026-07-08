"""Tests for pre-trade risk and compliance checks."""

import pytest

from src.oms import (
    Order,
    OrderType,
    Portfolio,
    PreTradeLimits,
    Side,
    pre_trade_check,
)


def _portfolio() -> Portfolio:
    return Portfolio(initial_cash=100_000.0)


def _buy(quantity: float = 100.0, **kwargs: object) -> Order:
    return Order(symbol="AAA", side=Side.BUY, quantity=quantity, **kwargs)  # type: ignore[arg-type]


def test_clean_order_passes() -> None:
    result = pre_trade_check(_buy(), _portfolio(), {"AAA": 100.0}, PreTradeLimits())
    assert result.ok
    assert result.violations == ()


def test_restricted_symbol_is_blocked() -> None:
    limits = PreTradeLimits(restricted_symbols=frozenset({"AAA"}))
    result = pre_trade_check(_buy(), _portfolio(), {"AAA": 100.0}, limits)
    assert not result.ok
    assert any("restricted" in v for v in result.violations)


def test_missing_mark_blocks_value_checks() -> None:
    result = pre_trade_check(_buy(), _portfolio(), {}, PreTradeLimits())
    assert not result.ok
    assert any("mark price" in v for v in result.violations)


def test_fat_finger_notional_cap() -> None:
    limits = PreTradeLimits(max_order_notional=5_000.0)
    result = pre_trade_check(_buy(quantity=100), _portfolio(), {"AAA": 100.0}, limits)
    assert not result.ok
    assert any("order notional" in v for v in result.violations)
    assert pre_trade_check(_buy(quantity=10), _portfolio(), {"AAA": 100.0}, limits).ok


def test_limit_price_values_the_order() -> None:
    # 100 shares limit 120 = 12k notional even though the mark is 100
    limits = PreTradeLimits(max_order_notional=11_000.0)
    order = _buy(quantity=100, order_type=OrderType.LIMIT, limit_price=120.0)
    result = pre_trade_check(order, _portfolio(), {"AAA": 100.0}, limits)
    assert not result.ok


def test_position_cap_counts_existing_holdings() -> None:
    portfolio = _portfolio()
    portfolio.record_fill("AAA", Side.BUY, 400, 100.0)
    limits = PreTradeLimits(max_position_notional=45_000.0)
    # 400 held + 100 new = 500 * 100 = 50k > 45k
    result = pre_trade_check(_buy(quantity=100), portfolio, {"AAA": 100.0}, limits)
    assert not result.ok
    assert any("position notional" in v for v in result.violations)
    # selling reduces the position and passes
    sell = Order(symbol="AAA", side=Side.SELL, quantity=100)
    assert pre_trade_check(sell, portfolio, {"AAA": 100.0}, limits).ok


def test_gross_leverage_cap_is_post_trade() -> None:
    limits = PreTradeLimits(max_gross_leverage=1.0)
    # 1500 * 100 = 150k gross on 100k equity -> 1.5x
    result = pre_trade_check(_buy(quantity=1_500), _portfolio(), {"AAA": 100.0}, limits)
    assert not result.ok
    assert any("gross leverage" in v for v in result.violations)
    assert pre_trade_check(_buy(quantity=900), _portfolio(), {"AAA": 100.0}, limits).ok


def test_price_collar_flags_far_limit_price() -> None:
    limits = PreTradeLimits(price_collar_pct=0.05)
    order = _buy(order_type=OrderType.LIMIT, limit_price=120.0)  # 20% above mark
    result = pre_trade_check(order, _portfolio(), {"AAA": 100.0}, limits)
    assert not result.ok
    assert any("collar" in v for v in result.violations)
    near = _buy(order_type=OrderType.LIMIT, limit_price=103.0)
    assert pre_trade_check(near, _portfolio(), {"AAA": 100.0}, limits).ok


def test_all_violations_reported_together() -> None:
    limits = PreTradeLimits(
        max_order_notional=1_000.0,
        max_gross_leverage=1.0,
        restricted_symbols=frozenset({"AAA"}),
    )
    result = pre_trade_check(_buy(quantity=2_000), _portfolio(), {"AAA": 100.0}, limits)
    assert not result.ok
    assert len(result.violations) == 3


def test_short_positions_count_toward_gross() -> None:
    portfolio = _portfolio()
    portfolio.record_fill("BBB", Side.SELL, 500, 100.0)  # 50k short
    limits = PreTradeLimits(max_gross_leverage=1.0)
    marks = {"AAA": 100.0, "BBB": 100.0}
    # 60k new long + 50k short = 110k gross on 100k equity
    result = pre_trade_check(_buy(quantity=600), portfolio, marks, limits)
    assert not result.ok


def test_non_positive_limit_raises() -> None:
    with pytest.raises(ValueError, match="max_order_notional"):
        PreTradeLimits(max_order_notional=0.0)
    with pytest.raises(ValueError, match="price_collar_pct"):
        PreTradeLimits(price_collar_pct=-0.1)
