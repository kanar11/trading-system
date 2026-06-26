"""Broker adapter abstraction.

A thin protocol that defines what *any* broker must support
(submit / cancel / positions / equity), with one concrete
:class:`PaperBroker` that delegates to the in-process OMS. This is
intentionally the *only* place that knows about external execution
venues; strategies submit orders through their event-driven
``Context`` and the engine talks to a broker.

The future-state roadmap:

    * ``InteractiveBrokersBroker``  — IB Gateway / TWS via ib_insync.
    * ``AlpacaBroker``              — Alpaca REST / streaming.
    * ``BinanceBroker``             — CCXT or python-binance.

All of them will subclass :class:`Broker` and implement the same six
methods that :class:`PaperBroker` does, so a strategy can move from
paper to live by swapping one constructor.
"""

from __future__ import annotations

import itertools
import logging
from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime

from src.oms import Order, OrderStatus, OrderType, Portfolio, Side

logger = logging.getLogger(__name__)


@dataclass
class BrokerFill:
    """A fill report returned by a Broker after submitting an order."""

    order_id: int
    symbol: str
    side: Side
    quantity: float
    price: float
    commission: float
    when: datetime


class Broker(ABC):
    """Abstract broker interface.

    All methods are synchronous for simplicity — async / streaming
    variants live one layer up (out of scope for this initial cut).
    """

    @abstractmethod
    def submit_order(self, order: Order) -> Order:
        """Submit an order. Returns the same Order with broker-assigned id + status."""

    @abstractmethod
    def cancel_order(self, order_id: int) -> bool:
        """Cancel an outstanding order. Returns True if cancelled, False if already terminal."""

    @abstractmethod
    def open_orders(self, symbol: str | None = None) -> list[Order]:
        """Return all currently-active orders, optionally filtered by symbol."""

    @abstractmethod
    def positions(self) -> Mapping[str, float]:
        """Return current per-symbol signed quantities."""

    @abstractmethod
    def equity(self, marks: Mapping[str, float]) -> float:
        """Total account equity at the given mark prices."""

    @abstractmethod
    def cash(self) -> float:
        """Free cash balance."""


class PaperBroker(Broker):
    """Simulated broker backed by an in-process :class:`Portfolio`.

    Fills are immediate at the supplied mark price (passed at submission
    time as ``mark_price``). This makes ``PaperBroker`` a clean
    bridge between strategy logic and the OMS without requiring a full
    bar-by-bar engine — useful for unit-testing strategies in isolation
    and for first-pass paper-trading harnesses.

    For end-to-end realistic simulation use :class:`EventEngine`
    instead, which models intrabar limit / stop matching.
    """

    def __init__(
        self,
        initial_cash: float = 100_000.0,
        commission_per_share: float = 0.0,
    ):
        self._portfolio = Portfolio(initial_cash=initial_cash)
        self._next_id = itertools.count(1)
        self._orders: dict[int, Order] = {}
        self.commission_per_share = commission_per_share
        self.fills: list[BrokerFill] = []

    # --- Broker interface -------------------------------------------------

    def submit_order(self, order: Order, mark_price: float | None = None) -> Order:
        """Submit and immediately attempt to fill ``order`` at ``mark_price``.

        Args:
            order: A constructed :class:`Order` (without an id).
            mark_price: Price to fill at. Required for MARKET orders;
                LIMIT orders check it against ``order.limit_price``.

        Returns:
            The same Order, mutated to the resulting status.
        """
        order.order_id = next(self._next_id)
        order.status = OrderStatus.WORKING
        order.created_at = datetime.utcnow()
        self._orders[order.order_id] = order

        if mark_price is None:
            # leave the order working — caller will need to invoke
            # ``poll(marks)`` to attempt a fill later.
            return order

        self._try_fill(order, mark_price)
        return order

    def cancel_order(self, order_id: int) -> bool:
        order = self._orders.get(order_id)
        if order is None or order.status.is_terminal:
            return False
        order.cancel()
        return True

    def open_orders(self, symbol: str | None = None) -> list[Order]:
        return [
            o
            for o in self._orders.values()
            if o.status.is_active and (symbol is None or o.symbol == symbol)
        ]

    def positions(self) -> Mapping[str, float]:
        return {s: p.quantity for s, p in self._portfolio.positions.items() if not p.is_flat}

    def equity(self, marks: Mapping[str, float]) -> float:
        return self._portfolio.equity(marks)

    def cash(self) -> float:
        return self._portfolio.cash

    # --- paper-specific helpers ------------------------------------------

    def poll(self, marks: Mapping[str, float]) -> list[BrokerFill]:
        """Re-evaluate all working orders against new ``marks``.

        Useful for stepping a paper-trading loop forward without a
        full event engine.
        """
        new_fills: list[BrokerFill] = []
        for order in list(self._orders.values()):
            if not order.status.is_active:
                continue
            mark = marks.get(order.symbol)
            if mark is None:
                continue
            fill = self._try_fill(order, mark)
            if fill is not None:
                new_fills.append(fill)
        return new_fills

    @property
    def portfolio(self) -> Portfolio:
        """Expose the underlying portfolio (read-only by convention)."""
        return self._portfolio

    # --- internals --------------------------------------------------------

    def _try_fill(self, order: Order, mark: float) -> BrokerFill | None:
        """Attempt to fill ``order`` at ``mark``. Returns a BrokerFill on success."""
        if order.order_type is OrderType.MARKET:
            fill_price = mark
        elif order.order_type is OrderType.LIMIT:
            limit = order.limit_price
            assert limit is not None  # guaranteed for LIMIT (Order.__post_init__)
            if (order.side is Side.BUY and mark <= limit) or (
                order.side is Side.SELL and mark >= limit
            ):
                fill_price = limit
            else:
                return None
        elif order.order_type is OrderType.STOP:
            stop = order.stop_price
            assert stop is not None  # guaranteed for STOP (Order.__post_init__)
            triggered = (order.side is Side.BUY and mark >= stop) or (
                order.side is Side.SELL and mark <= stop
            )
            if not triggered:
                return None
            fill_price = mark
        else:
            order.reject(f"unsupported order type {order.order_type}")
            return None

        qty = order.remaining_quantity
        commission = qty * self.commission_per_share
        order.record_fill(qty, fill_price)
        self._portfolio.record_fill(
            symbol=order.symbol,
            side=order.side,
            quantity=qty,
            price=fill_price,
            commission=commission,
        )
        assert order.order_id is not None  # assigned by submit() before matching
        fill = BrokerFill(
            order_id=order.order_id,
            symbol=order.symbol,
            side=order.side,
            quantity=qty,
            price=fill_price,
            commission=commission,
            when=datetime.utcnow(),
        )
        self.fills.append(fill)
        return fill
