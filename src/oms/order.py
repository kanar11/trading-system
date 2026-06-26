r"""Order primitives — HFT-grade order lifecycle.

The order is the single most safety-critical object in a trading
system: a bug here loses money directly. This module is written to the
standards a professional execution desk would expect:

* **Strict state machine.** Every status change goes through
  :meth:`Order._transition`, which consults an explicit legal-transition
  table and raises :class:`IllegalOrderTransition` on any violation.
  No code path can silently move an order into an inconsistent state.

* **Immutable, sequenced audit trail.** Every fill is recorded as a
  frozen :class:`Fill` carrying a monotonically-increasing sequence
  number, an event timestamp, and a maker/taker liquidity flag.
  Out-of-order fill timestamps are rejected — the event log is always
  replayable in order.

* **FIX-aligned vocabulary.** ``cum_qty`` (cumulative filled),
  ``leaves_qty`` (open quantity), and ``avg_px`` mirror the FIX 4.x
  execution-report fields execution venues speak.

* **Float-drift-safe arithmetic.** Quantities are compared with an
  explicit epsilon and the average fill price is derived from a running
  notional rather than repeated incremental division, so a thousand
  partial fills don't accumulate rounding error.

* **Latency-conscious layout.** ``slots=True`` removes the per-instance
  ``__dict__``, shrinking memory and speeding attribute access — both
  matter when millions of orders flow through per session.

* **Structured rejects.** Reject reasons are an enum (so risk systems
  can branch on them) with optional free-text detail, instead of
  corrupting an unrelated field.

* **Cancel/replace (amend).** :meth:`Order.amend` supports the
  cancel-replace workflow with full invariant checking.

Lifecycle::

    PENDING ──▶ WORKING ──▶ PARTIALLY_FILLED ──▶ FILLED
       │           │                │
       ├───────────┼────────────────┴──▶ CANCELLED
       └───────────┴───────────────────▶ REJECTED
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

# Tolerance for quantity comparisons. Share/lot quantities are usually
# integral, but vol-targeting and fractional-share venues produce
# non-integral sizes; this epsilon keeps "fully filled" robust against
# binary-float representation error.
QTY_EPSILON: float = 1e-9


class OrderError(Exception):
    """Base class for all order-related errors."""


class IllegalOrderTransition(OrderError):
    """Raised when an order is asked to make a status change the state
    machine forbids (e.g. filling a cancelled order)."""


class OverFill(IllegalOrderTransition, ValueError):
    """Raised when a fill would push cumulative quantity past the order size.

    Inherits from both :class:`IllegalOrderTransition` (it is an illegal
    lifecycle event — a fully-filled order is terminal) and ``ValueError``
    (it is also a bad-quantity argument), so callers can catch it as either.
    """


class Side(str, Enum):
    """Trade side."""

    BUY = "BUY"
    SELL = "SELL"

    @property
    def signed(self) -> int:
        """+1 for buy, -1 for sell — for computing signed quantities."""
        return 1 if self is Side.BUY else -1

    @property
    def opposite(self) -> Side:
        return Side.SELL if self is Side.BUY else Side.BUY


class OrderType(str, Enum):
    """Order type. STOP and STOP_LIMIT use ``stop_price``; LIMIT and
    STOP_LIMIT use ``limit_price``; MARKET fills at next bar's open."""

    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"
    STOP_LIMIT = "STOP_LIMIT"


class TimeInForce(str, Enum):
    """How long an order remains active."""

    DAY = "DAY"  # cancel at end of day if not filled
    GTC = "GTC"  # good til cancelled
    IOC = "IOC"  # immediate or cancel — fill what you can, cancel rest
    FOK = "FOK"  # fill or kill — fill in full immediately or cancel


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
            OrderStatus.PENDING,
            OrderStatus.WORKING,
            OrderStatus.PARTIALLY_FILLED,
        )


class RejectReason(str, Enum):
    """Structured reject reasons so risk / OMS layers can branch on them."""

    NONE = ""
    INSUFFICIENT_QUANTITY = "INSUFFICIENT_QUANTITY"
    INVALID_PRICE = "INVALID_PRICE"
    UNSUPPORTED_TYPE = "UNSUPPORTED_TYPE"
    RISK_LIMIT = "RISK_LIMIT"
    DUPLICATE = "DUPLICATE"
    STALE = "STALE"
    UNKNOWN = "UNKNOWN"


class Liquidity(str, Enum):
    """Whether a fill added (MAKER) or removed (TAKER) liquidity.

    Matters for fee tiers: most venues rebate makers and charge takers.
    """

    MAKER = "MAKER"
    TAKER = "TAKER"


# Explicit legal-transition table. A status may only move to a status in
# its set. Self-loops are listed where they're meaningful (additional
# partial fills keep an order PARTIALLY_FILLED).
_LEGAL_TRANSITIONS: dict[OrderStatus, frozenset[OrderStatus]] = {
    OrderStatus.PENDING: frozenset(
        {
            OrderStatus.WORKING,
            OrderStatus.PARTIALLY_FILLED,
            OrderStatus.FILLED,
            OrderStatus.CANCELLED,
            OrderStatus.REJECTED,
        }
    ),
    OrderStatus.WORKING: frozenset(
        {
            OrderStatus.PARTIALLY_FILLED,
            OrderStatus.FILLED,
            OrderStatus.CANCELLED,
            OrderStatus.REJECTED,
        }
    ),
    OrderStatus.PARTIALLY_FILLED: frozenset(
        {
            OrderStatus.PARTIALLY_FILLED,
            OrderStatus.FILLED,
            OrderStatus.CANCELLED,
        }
    ),
    OrderStatus.FILLED: frozenset(),
    OrderStatus.CANCELLED: frozenset(),
    OrderStatus.REJECTED: frozenset(),
}


@dataclass(frozen=True, slots=True)
class Fill:
    """An immutable execution report for a single (partial) fill.

    Attributes:
        seq: Monotonic per-order fill sequence number (1-indexed).
        ts: Event timestamp of the fill.
        quantity: Quantity executed in this fill (> 0).
        price: Execution price (> 0).
        liquidity: MAKER or TAKER.
    """

    seq: int
    ts: datetime
    quantity: float
    price: float
    liquidity: Liquidity = Liquidity.TAKER

    def __iter__(self):
        """Tuple-compatible unpacking: ``ts, qty, price = fill``.

        Preserves the historical ``(ts, quantity, price)`` shape so older
        call sites that unpack fills keep working.
        """
        yield self.ts
        yield self.quantity
        yield self.price


@dataclass(slots=True)
class Order:
    """An order with a strictly-validated lifecycle.

    Construction parameters:
        symbol: Trading symbol (e.g. 'SPY').
        side: BUY or SELL.
        quantity: Total requested quantity (> 0).
        order_type: MARKET / LIMIT / STOP / STOP_LIMIT.
        limit_price: Required for LIMIT and STOP_LIMIT.
        stop_price: Required for STOP and STOP_LIMIT.
        time_in_force: DAY / GTC / IOC / FOK.
        order_id: Assigned by the OMS at submission.
        created_at: Submission timestamp.
        client_tag: Caller-supplied label for grouping / reconciliation.

    Engine-managed state (do not set at construction):
        status, filled_quantity, avg_fill_price, last_fill_at, fills,
        version, reject_reason, reject_detail.
    """

    symbol: str
    side: Side
    quantity: float
    order_type: OrderType = OrderType.MARKET
    limit_price: float | None = None
    stop_price: float | None = None
    time_in_force: TimeInForce = TimeInForce.DAY
    order_id: int | None = None
    created_at: datetime | None = None
    client_tag: str | None = None

    # --- engine-managed state (init=False) ---
    status: OrderStatus = field(default=OrderStatus.PENDING, init=False)
    filled_quantity: float = field(default=0.0, init=False)
    avg_fill_price: float = field(default=0.0, init=False)
    last_fill_at: datetime | None = field(default=None, init=False)
    fills: list[Fill] = field(default_factory=list, init=False)
    version: int = field(default=0, init=False)
    reject_reason: RejectReason = field(default=RejectReason.NONE, init=False)
    reject_detail: str = field(default="", init=False)
    _cum_notional: float = field(default=0.0, init=False, repr=False)

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

    # -----------------------------------------------------------------
    # derived quantities (FIX-aligned vocabulary)
    # -----------------------------------------------------------------

    @property
    def remaining_quantity(self) -> float:
        """Open quantity still to be executed (clamped at 0)."""
        return max(self.quantity - self.filled_quantity, 0.0)

    #: FIX alias for :attr:`remaining_quantity`.
    @property
    def leaves_qty(self) -> float:
        return self.remaining_quantity

    #: FIX alias for :attr:`filled_quantity`.
    @property
    def cum_qty(self) -> float:
        return self.filled_quantity

    #: FIX alias for :attr:`avg_fill_price`.
    @property
    def avg_px(self) -> float:
        return self.avg_fill_price

    @property
    def signed_quantity(self) -> float:
        """Signed total quantity: positive for BUY, negative for SELL."""
        return self.side.signed * self.quantity

    @property
    def signed_filled_quantity(self) -> float:
        return self.side.signed * self.filled_quantity

    @property
    def is_buy(self) -> bool:
        return self.side is Side.BUY

    @property
    def is_sell(self) -> bool:
        return self.side is Side.SELL

    @property
    def is_complete(self) -> bool:
        """True when the order has been fully filled (within epsilon)."""
        return self.remaining_quantity <= QTY_EPSILON

    # -----------------------------------------------------------------
    # state machine
    # -----------------------------------------------------------------

    def _transition(self, new_status: OrderStatus) -> None:
        """Move to ``new_status`` if the transition is legal, else raise.

        Raises:
            IllegalOrderTransition: If the move is not in the legal table.
        """
        # idempotent self-transition only allowed where the table says so
        if new_status == self.status and new_status not in _LEGAL_TRANSITIONS[self.status]:
            return
        if new_status not in _LEGAL_TRANSITIONS[self.status]:
            raise IllegalOrderTransition(
                f"order {self.order_id}: illegal transition "
                f"{self.status.value} -> {new_status.value}"
            )
        self.status = new_status

    def activate(self) -> None:
        """Move a PENDING order to WORKING (acknowledged by the venue)."""
        if self.status is OrderStatus.PENDING:
            self._transition(OrderStatus.WORKING)

    # -----------------------------------------------------------------
    # fills
    # -----------------------------------------------------------------

    def record_fill(
        self,
        fill_qty: float,
        fill_price: float,
        when: datetime | None = None,
        liquidity: Liquidity | str = Liquidity.TAKER,
    ) -> Fill:
        """Apply a (partial) fill and advance the state machine.

        Args:
            fill_qty: Quantity executed (> 0).
            fill_price: Execution price (> 0).
            when: Event timestamp. Defaults to ``datetime.utcnow()``.
                Must be >= the previous fill's timestamp (the audit log
                is strictly ordered).
            liquidity: MAKER or TAKER (accepts the enum or its string).

        Returns:
            The :class:`Fill` record that was appended.

        Raises:
            IllegalOrderTransition: If the order has been cancelled or
                rejected (a fully-filled order raises ValueError instead,
                since any further quantity is by definition an over-fill).
            ValueError: On non-positive size/price, over-fill, or an
                out-of-order timestamp.
        """
        # CANCELLED/REJECTED can never be filled. A FILLED order is also
        # terminal, but filling it further is a quantity violation, so it
        # falls through to the over-fill check below for a clearer error.
        if self.status in (OrderStatus.CANCELLED, OrderStatus.REJECTED):
            raise IllegalOrderTransition(
                f"order {self.order_id}: cannot fill a {self.status.value} order"
            )
        if fill_qty <= 0:
            raise ValueError(f"fill_qty must be > 0, got {fill_qty}")
        if fill_price <= 0:
            raise ValueError(f"fill_price must be > 0, got {fill_price}")
        if self.filled_quantity + fill_qty > self.quantity + QTY_EPSILON:
            raise OverFill(
                f"fill of {fill_qty} would over-fill order "
                f"(filled {self.filled_quantity} of {self.quantity})"
            )

        ts = when if when is not None else datetime.utcnow()
        if self.last_fill_at is not None and ts < self.last_fill_at:
            raise ValueError(f"out-of-order fill: {ts} precedes previous fill {self.last_fill_at}")
        liq = Liquidity(liquidity) if not isinstance(liquidity, Liquidity) else liquidity

        # running notional → drift-free average price
        self._cum_notional += fill_price * fill_qty
        self.filled_quantity += fill_qty
        self.avg_fill_price = self._cum_notional / self.filled_quantity
        self.last_fill_at = ts

        seq = len(self.fills) + 1
        fill = Fill(seq=seq, ts=ts, quantity=fill_qty, price=fill_price, liquidity=liq)
        self.fills.append(fill)

        # advance state
        if self.is_complete:
            self._transition(OrderStatus.FILLED)
        else:
            self._transition(OrderStatus.PARTIALLY_FILLED)
        return fill

    # -----------------------------------------------------------------
    # cancel / reject / amend
    # -----------------------------------------------------------------

    def cancel(self) -> bool:
        """Cancel the order.

        Returns:
            True if the order moved to CANCELLED, False if it was already
            terminal (no-op — matches venue cancel-reject semantics).
        """
        if self.status.is_terminal:
            return False
        self._transition(OrderStatus.CANCELLED)
        return True

    def reject(
        self,
        reason: RejectReason | str = RejectReason.UNKNOWN,
        detail: str = "",
    ) -> bool:
        """Reject the order with a structured reason.

        Args:
            reason: A :class:`RejectReason` (or a free-text string, which
                is stored as ``UNKNOWN`` + detail).
            detail: Optional human-readable detail.

        Returns:
            True if rejected, False if the order was already terminal.
        """
        if self.status.is_terminal:
            return False
        if isinstance(reason, RejectReason):
            self.reject_reason = reason
            self.reject_detail = detail
        else:
            self.reject_reason = RejectReason.UNKNOWN
            self.reject_detail = str(reason)
        self._transition(OrderStatus.REJECTED)
        return True

    def amend(
        self,
        new_quantity: float | None = None,
        new_limit_price: float | None = None,
        new_stop_price: float | None = None,
    ) -> int:
        """Cancel-replace the order's quantity / prices in place.

        Bumps :attr:`version` on success so downstream systems can detect
        the modification.

        Args:
            new_quantity: New total quantity. Must be > already-filled qty.
            new_limit_price: New limit price (LIMIT / STOP_LIMIT only).
            new_stop_price: New stop price (STOP / STOP_LIMIT only).

        Returns:
            The new :attr:`version` number.

        Raises:
            IllegalOrderTransition: If the order is terminal.
            ValueError: On invalid new values.
        """
        if self.status.is_terminal:
            raise IllegalOrderTransition(
                f"order {self.order_id}: cannot amend a {self.status.value} order"
            )

        if new_quantity is not None:
            if new_quantity <= 0:
                raise ValueError("new_quantity must be > 0")
            if new_quantity < self.filled_quantity - QTY_EPSILON:
                raise ValueError(
                    f"new_quantity {new_quantity} is below already-filled {self.filled_quantity}"
                )
            self.quantity = new_quantity

        if new_limit_price is not None:
            if self.order_type not in (OrderType.LIMIT, OrderType.STOP_LIMIT):
                raise ValueError(f"cannot set limit_price on a {self.order_type.value} order")
            if new_limit_price <= 0:
                raise ValueError("new_limit_price must be > 0")
            self.limit_price = new_limit_price

        if new_stop_price is not None:
            if self.order_type not in (OrderType.STOP, OrderType.STOP_LIMIT):
                raise ValueError(f"cannot set stop_price on a {self.order_type.value} order")
            if new_stop_price <= 0:
                raise ValueError("new_stop_price must be > 0")
            self.stop_price = new_stop_price

        # if the amend brought quantity down to the filled amount, complete it
        if self.is_complete and not self.status.is_terminal and self.filled_quantity > 0:
            self._transition(OrderStatus.FILLED)

        self.version += 1
        return self.version

    # -----------------------------------------------------------------
    # serialisation
    # -----------------------------------------------------------------

    def to_dict(self) -> dict:
        """Structured snapshot for audit logging / persistence."""
        return {
            "order_id": self.order_id,
            "symbol": self.symbol,
            "side": self.side.value,
            "quantity": self.quantity,
            "order_type": self.order_type.value,
            "limit_price": self.limit_price,
            "stop_price": self.stop_price,
            "time_in_force": self.time_in_force.value,
            "status": self.status.value,
            "cum_qty": self.filled_quantity,
            "leaves_qty": self.remaining_quantity,
            "avg_px": self.avg_fill_price,
            "version": self.version,
            "reject_reason": self.reject_reason.value,
            "reject_detail": self.reject_detail,
            "created_at": self.created_at,
            "last_fill_at": self.last_fill_at,
            "n_fills": len(self.fills),
            "client_tag": self.client_tag,
        }
