"""Rebalance order generation: target weights → executable orders.

The allocation layer thinks in *weights* (:mod:`src.portfolio` optimisers,
:func:`src.strategy.dual_momentum.dual_momentum_strategy`); the OMS thinks
in *orders*. This module bridges the two: given a live
:class:`~src.oms.portfolio.Portfolio`, a target-weight map and mark prices,
it emits the BUY/SELL quantities that move the book from its current
holdings to the targets. Held symbols missing from the targets are closed
(implicit target 0); negative weights are short targets.

Dust control is built in: deltas below ``min_notional`` are skipped and
quantities can be rounded down to a ``lot_size`` grid, so the generator
does not spray micro-orders on every drifted position. Inputs are never
mutated — this is a pure planning step; execution (and therefore fees,
see :mod:`src.oms.fees`) stays with the caller.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

from src.oms.order import Side
from src.oms.portfolio import Portfolio


@dataclass(frozen=True)
class RebalanceOrder:
    """One planned order of a rebalance.

    Attributes:
        symbol: Trading symbol.
        side: BUY or SELL.
        quantity: Unsigned quantity to trade (> 0).
        notional: Unsigned mark-price notional of the order.
    """

    symbol: str
    side: Side
    quantity: float
    notional: float


def rebalance_orders(
    portfolio: Portfolio,
    target_weights: Mapping[str, float],
    marks: Mapping[str, float],
    min_notional: float = 0.0,
    lot_size: float = 0.0,
) -> list[RebalanceOrder]:
    """Plan the orders that move ``portfolio`` to ``target_weights``.

    For every symbol in the targets or currently held: the target position
    value is ``weight * equity``; the difference to the current marked value
    is converted to a quantity at the mark price. Symbols held but absent
    from ``target_weights`` are treated as target 0 and closed.

    Args:
        portfolio: Current book (not mutated).
        target_weights: ``{symbol: weight}`` in fraction-of-equity units;
            weights may be negative (shorts) and need not sum to 1 (the
            remainder stays in cash).
        marks: ``{symbol: mark_price}`` covering every symbol involved.
        min_notional: Skip orders below this mark-price notional.
        lot_size: If > 0, round quantities *down* to this lot grid (e.g.
            1.0 for whole shares); orders rounding to zero are skipped.

    Returns:
        Planned orders sorted by symbol (deterministic).

    Raises:
        ValueError: If equity is non-positive, a weight is not finite, a
            needed mark is missing or non-positive, or a tolerance is
            negative.
    """
    if min_notional < 0:
        raise ValueError(f"min_notional must be >= 0, got {min_notional}.")
    if lot_size < 0:
        raise ValueError(f"lot_size must be >= 0, got {lot_size}.")
    for symbol, weight in target_weights.items():
        if not math.isfinite(weight):
            raise ValueError(f"weight for {symbol!r} must be finite, got {weight}.")

    held = {s for s, p in portfolio.positions.items() if not p.is_flat}
    symbols = sorted(held | set(target_weights))

    missing = [s for s in symbols if s not in marks]
    if missing:
        raise ValueError(f"missing mark prices for {missing}.")
    non_positive = [s for s in symbols if marks[s] <= 0]
    if non_positive:
        raise ValueError(f"mark prices must be > 0, got non-positive for {non_positive}.")

    equity = portfolio.equity(marks)
    if equity <= 0:
        raise ValueError(f"portfolio equity must be > 0, got {equity}.")

    orders: list[RebalanceOrder] = []
    for symbol in symbols:
        mark = float(marks[symbol])
        position = portfolio.positions.get(symbol)
        current_qty = position.quantity if position is not None else 0.0

        target_qty = target_weights.get(symbol, 0.0) * equity / mark
        delta_qty = target_qty - current_qty

        quantity = abs(delta_qty)
        if lot_size > 0:
            quantity = math.floor(quantity / lot_size) * lot_size
        notional = quantity * mark
        if quantity <= 0 or notional < min_notional:
            continue

        side = Side.BUY if delta_qty > 0 else Side.SELL
        orders.append(
            RebalanceOrder(symbol=symbol, side=side, quantity=quantity, notional=notional)
        )
    return orders
