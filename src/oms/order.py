r"""Order primitives used by the OMS and event-driven engine.

These are *pure-Python* dataclasses with no pandas / numpy dependency
so they're cheap to construct, serialise, and equality-compare in
tests. The event engine is responsible for transitioning an order
through its lifecycle:

    PENDING -> WORKING -> [PARTIALLY_FILLED] -> FILLED
                       \                     \
                        -> CANCELLED          -> CANCELLED
                       \
                        -> REJECTED
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional


class Side(str, Enum):
    """Trade side."""
    BUY = "BUY"
    SELL = "SELL"

    @property
    def signed(self) -> int:
        """+1 for buy, -1 for sell — useful when computing signed quantities."""
        return 1 if self is Side.BUY else -1


class OrderType(str, Enum):
    """Order type. STOP and STOP_LIMIT use ``stop_price``; LIMIT uses
    ``limit_price``; MARKET fills at next bar's open."""
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"
    STOP_LIMIT = "STOP_LIMIT"


class TimeInForce(str, Enum):
    """How long an order remains active."""
    DAY = "DAY"      # cancel at end of day if not filled
    GTC = "GTC"      # good til cancelled
    IOC = "IOC"      # immediate or cancel — fills what it can, cancels rest
    FOK = "FOK"      # fill or kill — fill in full immediately or cancel


class OrderStatus(str, Enum):
    """Lifecycle state of an order."""
    PENDING = "PENDING"
    WORKING = "WORKING"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"

    @property
    def is_terminal(self) -> bool:
        return self in (OrderStatus.FILLED, OrderStatus.CANCELLED, OrderStatus.REJECTED)

    @property
    def is_active(self) -> bool:
        return self in (
            OrderStatus.PENDING, OrderStatus.WORKING, OrderStatus.PARTIALLY_FILLED,
        )


@dataclass
class Order:
    """An order submitted to the OMS.

    Attributes:
        order_id: Unique identifier assigned by the OMS at submission.
        symbol: Trading symbol (e.g. 'SPY').
        side: BUY or SELL.
        quantity: Total requested quantity (positive).
        order_type: MARKET, LIMIT, STOP or STOP_LIMIT.
        limit_price: Price for LIMIT and STOP_LIMIT orders.
        stop_price: Trigger price for STOP and STOP_LIMIT orders.
        time_in_force: DAY / GTC / IOC / FOK.
        created_at: Wall-clock or simulation timestamp at submission.
        status: Current lifecycle status.
        filled_quantity: Cumulative quantity filled so far.
        avg_fill_price: Volume-weighted average fill price.
        last_fill_at: Timestamp of the most recent fill (None if no fills).
        client_tag: Optional caller-supplied label for grouping orders.
    """

    symbol: str
    side: Side
    quantity: float
    order_type: OrderType = OrderType.MARKET
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    time_in_force: TimeInForce = TimeInForce.DAY
    order_id: Optional[int] = None
    created_at: Optional[datetime] = None
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: float = 0.0
    avg_fill_price: float = 0.0
    last_fill_at: Optional[datetime] = None
    client_tag: Optional[str] = None
    fills: list[tuple[datetime, float, float]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.quantity <= 0:
            raise ValueError(f"quantity must be > 0, got {self.quantity}")
        if self.order_type in (OrderType.LIMIT, OrderType.STOP_LIMIT) and self.limit_price is None:
            raise ValueError(f"{self.order_type.value} order requires limit_price")
        if self.order_type in (OrderType.STOP, OrderType.STOP_LIMIT) and self.stop_price is None:
            raise ValueError(f"{self.order_type.value} order requires stop_price")
        if self.limit_price is not None and self.limit_price <= 0:
            raise ValueError("limit_price must be > 0")
        if self.stop_price is not None and self.stop_price <= 0:
            raise ValueError("stop_price must be > 0")

    @property
    def remaining_quantity(self) -> float:
        """Quantity still outstanding (not yet filled or cancelled)."""
        return max(self.quantity - self.filled_quantity, 0.0)

    @property
    def signed_quantity(self) -> float:
        """Signed quantity: positive for BUY, negative for SELL."""
        return self.side.signed * self.quantity

    @property
    def signed_filled_quantity(self) -> float:
        return self.side.signed * self.filled_quantity

    def record_fill(self, fill_qty: float, fill_price: float, when: Optional[datetime] = None) -> None:
        """Apply a (partial) fill, updating avg price and status.

        Args:
            fill_qty: Quantity filled in this event (must be > 0).
            fill_price: Trade price for this fill.
            when: Timestamp of the fill. Defaults to ``datetime.utcnow()``.

        Raises:
            ValueError: If the fill would over-fill the order or on bad inputs.
        """
        if fill_qty <= 0:
            raise ValueError(f"fill_qty must be > 0, got {fill_qty}")
        if fill_price <= 0:
            raise ValueError(f"fill_price must be > 0, got {fill_price}")
        if self.filled_quantity + fill_qty > self.quantity + 1e-12:
            raise ValueError(
                f"fill of {fill_qty} would over-fill order "
                f"(filled {self.filled_quantity} of {self.quantity})"
            )

        ts = when if when is not None else datetime.utcnow()
        # weighted-avg price
        prev_notional = self.avg_fill_price * self.filled_quantity
        new_notional = prev_notional + fill_price * fill_qty
        self.filled_quantity += fill_qty
        self.avg_fill_price = new_notional / self.filled_quantity
        self.last_fill_at = ts
        self.fills.append((ts, fill_qty, fill_price))

        # transition
        if abs(self.filled_quantity - self.quantity) < 1e-9:
            self.status = OrderStatus.FILLED
        else:
            self.status = OrderStatus.PARTIALLY_FILLED

    def cancel(self) -> None:
        """Cancel the order. No-op if already terminal."""
        if not self.status.is_terminal:
            self.status = OrderStatus.CANCELLED

    def reject(self, reason: str = "") -> None:
        """Mark the order as rejected. No-op if already terminal."""
        if not self.status.is_terminal:
            self.status = OrderStatus.REJECTED
            if reason:
                self.client_tag = (self.client_tag or "") + f" REJECTED: {reason}"
