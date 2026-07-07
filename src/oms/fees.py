"""Commission and fee schedules for order execution.

Brokers charge in three common shapes — a flat ticket charge, a per-share
rate, and a percentage of notional — usually floored by a minimum and
sometimes capped by a maximum. :class:`FeeSchedule` composes all of them in
one immutable value object so backtests and TCA can price executions
consistently, and ships factory presets for the typical retail plans.

Commissions are *per execution*: :meth:`FeeSchedule.commission` prices one
(quantity, price) execution, :func:`total_commission` sums a sequence of
:class:`~src.oms.order.Fill` objects, applying the schedule to each fill
independently (a broker that charges its minimum once per *order* will
therefore charge slightly less than this on partially-filled orders).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from src.oms.order import Fill


@dataclass(frozen=True)
class FeeSchedule:
    """An immutable broker commission schedule.

    The commission for an execution of ``quantity`` at ``price`` is::

        fee = per_order + per_share * quantity + pct_notional * quantity * price
        fee = max(fee, minimum)
        fee = min(fee, maximum)      # only when a maximum is set

    A zero-quantity execution is always free (no trade, no ticket charge).

    Attributes:
        per_order: Flat charge per execution.
        per_share: Charge per share/contract/unit.
        pct_notional: Charge as a fraction of traded notional (0.0005 = 5 bps).
        minimum: Floor applied after the components are summed.
        maximum: Optional cap applied last (``None`` = uncapped).
    """

    per_order: float = 0.0
    per_share: float = 0.0
    pct_notional: float = 0.0
    minimum: float = 0.0
    maximum: float | None = None

    def __post_init__(self) -> None:
        for name in ("per_order", "per_share", "pct_notional", "minimum"):
            value = float(getattr(self, name))
            if value < 0:
                raise ValueError(f"{name} must be >= 0, got {value}.")
        if self.maximum is not None:
            if self.maximum < 0:
                raise ValueError(f"maximum must be >= 0, got {self.maximum}.")
            if self.maximum < self.minimum:
                raise ValueError(
                    f"maximum ({self.maximum}) cannot be below minimum ({self.minimum})."
                )

    @classmethod
    def zero(cls) -> FeeSchedule:
        """Commission-free schedule (the modern US retail default)."""
        return cls()

    @classmethod
    def per_share_plan(cls, rate: float = 0.005, minimum: float = 1.0) -> FeeSchedule:
        """US per-share pricing (e.g. IBKR-fixed style: $0.005/share, $1 min)."""
        return cls(per_share=rate, minimum=minimum)

    @classmethod
    def bps_plan(cls, bps: float = 5.0, minimum: float = 0.0) -> FeeSchedule:
        """Percent-of-notional pricing quoted in basis points of trade value."""
        return cls(pct_notional=bps / 10_000.0, minimum=minimum)

    def commission(self, quantity: float, price: float) -> float:
        """Commission for a single execution of ``quantity`` at ``price``.

        Args:
            quantity: Executed quantity (>= 0; sign-free — pass ``abs()`` for
                sells).
            price: Execution price (>= 0).

        Returns:
            The commission amount (>= 0); 0.0 for a zero-quantity execution.

        Raises:
            ValueError: If ``quantity`` or ``price`` is negative.
        """
        if quantity < 0:
            raise ValueError(f"quantity must be >= 0, got {quantity}.")
        if price < 0:
            raise ValueError(f"price must be >= 0, got {price}.")
        if quantity == 0:
            return 0.0

        fee = self.per_order + self.per_share * quantity + self.pct_notional * quantity * price
        fee = max(fee, self.minimum)
        if self.maximum is not None:
            fee = min(fee, self.maximum)
        return float(fee)

    def fill_commission(self, fill: Fill) -> float:
        """Commission for one :class:`~src.oms.order.Fill`."""
        return self.commission(fill.quantity, fill.price)


def total_commission(fills: Sequence[Fill], schedule: FeeSchedule) -> float:
    """Total commission over ``fills``, pricing each fill independently.

    Args:
        fills: Fills to price (e.g. ``order.fills``).
        schedule: The commission schedule to apply.

    Returns:
        Sum of per-fill commissions (0.0 for an empty sequence).
    """
    return float(sum(schedule.fill_commission(f) for f in fills))
