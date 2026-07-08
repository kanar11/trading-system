"""Pre-trade risk and compliance checks.

Every institutional OMS gates outgoing orders through pre-trade checks
before they touch a venue: fat-finger notional caps, per-symbol position
limits, portfolio leverage caps, price collars against the current mark and
a restricted-trading list. This module is that gate in library form:
:func:`pre_trade_check` evaluates an :class:`~src.oms.order.Order` against a
:class:`PreTradeLimits` policy in the context of the current
:class:`~src.oms.portfolio.Portfolio` and returns *all* violations (not just
the first), so a rejected order can be repaired in one round trip.

The check is pure — nothing is mutated and no order state is transitioned;
wiring a rejection into ``Order.reject`` stays with the caller. Every limit
is optional (``None`` = not enforced), so a policy can start with a single
fat-finger cap and grow.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

from src.oms.order import Order, OrderType
from src.oms.portfolio import Portfolio


@dataclass(frozen=True)
class PreTradeLimits:
    """Pre-trade policy; ``None`` disables a given limit.

    Attributes:
        max_order_notional: Fat-finger cap on a single order's notional.
        max_position_notional: Cap on the post-trade absolute position
            value per symbol.
        max_gross_leverage: Cap on post-trade gross exposure / equity.
        price_collar_pct: Maximum fractional deviation of an order's
            limit/stop price from the current mark (0.05 = 5%).
        restricted_symbols: Symbols that must not be traded at all.
    """

    max_order_notional: float | None = None
    max_position_notional: float | None = None
    max_gross_leverage: float | None = None
    price_collar_pct: float | None = None
    restricted_symbols: frozenset[str] = field(default_factory=frozenset)

    def __post_init__(self) -> None:
        for name in (
            "max_order_notional",
            "max_position_notional",
            "max_gross_leverage",
            "price_collar_pct",
        ):
            value = getattr(self, name)
            if value is not None and value <= 0:
                raise ValueError(f"{name} must be > 0 when set, got {value}.")


@dataclass(frozen=True)
class CheckResult:
    """Outcome of a pre-trade check.

    Attributes:
        ok: True when no limit was violated.
        violations: Human-readable description of every violated limit.
    """

    ok: bool
    violations: tuple[str, ...]


def pre_trade_check(
    order: Order,
    portfolio: Portfolio,
    marks: Mapping[str, float],
    limits: PreTradeLimits,
) -> CheckResult:
    """Evaluate ``order`` against ``limits`` in the current portfolio context.

    Notional checks value the order at its limit price when it has one
    (the worst price it can fill at) and at the mark otherwise. Gross
    leverage is computed post-trade, assuming the full quantity fills;
    positions whose symbol has no mark are ignored there, matching
    :meth:`Portfolio.gross_exposure`.

    Args:
        order: The candidate order (not mutated).
        portfolio: Current book (not mutated).
        marks: ``{symbol: mark_price}``; the order's symbol must be marked
            for value-based checks to run.
        limits: The policy to enforce.

    Returns:
        A :class:`CheckResult`; ``ok`` is False when any check failed and
        ``violations`` lists all of them.
    """
    violations: list[str] = []

    if order.symbol in limits.restricted_symbols:
        violations.append(f"{order.symbol} is on the restricted list")

    mark = marks.get(order.symbol)
    if mark is None or mark <= 0:
        violations.append(f"no valid mark price for {order.symbol}; value checks impossible")
        return CheckResult(ok=not violations, violations=tuple(violations))

    if limits.price_collar_pct is not None:
        for label, price in (("limit", order.limit_price), ("stop", order.stop_price)):
            if price is not None:
                deviation = abs(price - mark) / mark
                if deviation > limits.price_collar_pct:
                    violations.append(
                        f"{label} price {price} deviates {deviation:.1%} from mark {mark} "
                        f"(collar {limits.price_collar_pct:.1%})"
                    )

    reference_price = order.limit_price if order.order_type is not OrderType.MARKET else None
    if reference_price is None:
        reference_price = mark

    order_notional = order.quantity * reference_price
    if limits.max_order_notional is not None and order_notional > limits.max_order_notional:
        violations.append(
            f"order notional {order_notional:,.2f} exceeds cap {limits.max_order_notional:,.2f}"
        )

    current_qty = 0.0
    position = portfolio.positions.get(order.symbol)
    if position is not None:
        current_qty = position.quantity
    post_qty = current_qty + order.side.signed * order.quantity

    if limits.max_position_notional is not None:
        post_position_notional = abs(post_qty) * mark
        if post_position_notional > limits.max_position_notional:
            violations.append(
                f"post-trade position notional {post_position_notional:,.2f} for "
                f"{order.symbol} exceeds cap {limits.max_position_notional:,.2f}"
            )

    if limits.max_gross_leverage is not None:
        equity = portfolio.equity(marks)
        if equity <= 0:
            violations.append(f"portfolio equity {equity:,.2f} is non-positive")
        else:
            gross = sum(
                abs(pos.quantity) * marks[sym]
                for sym, pos in portfolio.positions.items()
                if sym in marks and sym != order.symbol
            )
            gross += abs(post_qty) * mark
            leverage = gross / equity
            if leverage > limits.max_gross_leverage:
                violations.append(
                    f"post-trade gross leverage {leverage:.2f}x exceeds cap "
                    f"{limits.max_gross_leverage:.2f}x"
                )

    return CheckResult(ok=not violations, violations=tuple(violations))
