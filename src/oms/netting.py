"""Order netting: collapse a batch of orders into minimal net trades.

When several sub-strategies (or an ensemble, or a multi-signal book) all
emit orders in the same names, sending them raw pays the round-trip cost
on both a buy and an offsetting sell of the same symbol. The institutional
fix is *netting*: aggregate the batch by symbol into the single net order
each name actually needs, so fully-offsetting intents never reach the
market and partially-offsetting ones trade only their residual.

:func:`net_orders` takes a batch of MARKET orders and returns one net
MARKET order per symbol with a non-zero net quantity (symbols that cancel
out produce nothing), plus the gross-vs-net volume reduction — the saving
netting bought. Priced orders (LIMIT/STOP) are rejected: netting across
different price levels is ill-defined and belongs to the router, not here.
Inputs are never mutated.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from src.oms.order import Order, OrderType, Side


@dataclass
class NettingResult:
    """Outcome of netting an order batch.

    Attributes:
        orders: Net MARKET orders, one per symbol with a non-zero net
            quantity, sorted by symbol (deterministic).
        gross_quantity: Sum of absolute quantities of the input batch.
        net_quantity: Sum of absolute quantities of the net orders.
        reduction: ``1 − net_quantity / gross_quantity`` in [0, 1] — the
            fraction of traded volume removed by offsetting (0 when the
            batch has no offsets, 1 when it nets flat).
    """

    orders: list[Order]
    gross_quantity: float
    net_quantity: float
    reduction: float


def net_orders(orders: Sequence[Order]) -> NettingResult:
    """Aggregate a batch of MARKET orders into minimal net trades.

    Args:
        orders: Orders to net. All must be ``OrderType.MARKET``; a symbol's
            net signed quantity is the sum of its members' signed
            quantities, and the net order's ``client_tag`` is ``"netted"``.

    Returns:
        A :class:`NettingResult`.

    Raises:
        ValueError: If any order is not a MARKET order.
    """
    net_by_symbol: dict[str, float] = {}
    gross = 0.0
    for order in orders:
        if order.order_type is not OrderType.MARKET:
            raise ValueError(
                f"net_orders only handles MARKET orders, got {order.order_type.value} "
                f"for {order.symbol}."
            )
        net_by_symbol[order.symbol] = net_by_symbol.get(order.symbol, 0.0) + order.signed_quantity
        gross += order.quantity

    result_orders: list[Order] = []
    net = 0.0
    for symbol in sorted(net_by_symbol):
        signed = net_by_symbol[symbol]
        if abs(signed) < 1e-12:
            continue  # fully offset — nothing to trade
        result_orders.append(
            Order(
                symbol=symbol,
                side=Side.BUY if signed > 0 else Side.SELL,
                quantity=abs(signed),
                order_type=OrderType.MARKET,
                client_tag="netted",
            )
        )
        net += abs(signed)

    reduction = 1.0 - net / gross if gross > 0 else 0.0
    return NettingResult(
        orders=result_orders,
        gross_quantity=gross,
        net_quantity=net,
        reduction=reduction,
    )
