"""Portfolio state aggregation.

Owns cash, the per-symbol :class:`Position` objects, and a running
equity-history series. The event engine calls ``record_fill`` to
update positions and ``mark_to_market`` to snapshot equity after each
bar.

This module is *bookkeeping only*: it does not enforce risk limits,
issue orders, or know about strategies. Compose it with the OMS /
event engine to get a full trading harness.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Mapping

from src.oms.order import Side
from src.oms.position import Position


@dataclass
class Portfolio:
    """A collection of positions + cash + equity history.

    Attributes:
        initial_cash: Starting cash balance.
        cash: Current free cash (decreases when buying, increases when selling).
        positions: ``{symbol: Position}``.
        equity_history: List of ``(timestamp, equity)`` tuples appended on
            each call to :meth:`mark_to_market`.
        fees_paid: Cumulative commission / slippage paid.
    """

    initial_cash: float = 100_000.0
    cash: float = field(init=False)
    positions: dict[str, Position] = field(default_factory=dict)
    equity_history: list[tuple[datetime, float]] = field(default_factory=list)
    fees_paid: float = 0.0

    def __post_init__(self) -> None:
        if self.initial_cash <= 0:
            raise ValueError("initial_cash must be > 0")
        self.cash = float(self.initial_cash)

    # ---------------------------------------------------------------
    # accessors
    # ---------------------------------------------------------------

    def get_position(self, symbol: str) -> Position:
        """Return the Position for ``symbol``, creating an empty one if needed."""
        if symbol not in self.positions:
            self.positions[symbol] = Position(symbol=symbol)
        return self.positions[symbol]

    def total_realized_pnl(self) -> float:
        return sum(p.realized_pnl for p in self.positions.values())

    def total_unrealized_pnl(self, marks: Mapping[str, float]) -> float:
        return sum(
            p.unrealized_pnl(marks[s])
            for s, p in self.positions.items()
            if s in marks
        )

    def gross_exposure(self, marks: Mapping[str, float]) -> float:
        """Sum of absolute market values across all positions."""
        return sum(
            abs(p.market_value(marks[s]))
            for s, p in self.positions.items()
            if s in marks
        )

    def net_exposure(self, marks: Mapping[str, float]) -> float:
        return sum(
            p.market_value(marks[s])
            for s, p in self.positions.items()
            if s in marks
        )

    def equity(self, marks: Mapping[str, float]) -> float:
        """Cash + sum of position market values."""
        return self.cash + self.net_exposure(marks)

    # ---------------------------------------------------------------
    # mutations
    # ---------------------------------------------------------------

    def record_fill(
        self,
        symbol: str,
        side: Side,
        quantity: float,
        price: float,
        commission: float = 0.0,
    ) -> float:
        """Apply a single fill to the portfolio.

        - Updates the Position (cost basis, realised PnL).
        - Decreases / increases cash by ``signed_quantity * price``.
        - Subtracts commission from cash, accumulates ``fees_paid``.

        Args:
            symbol: Trading symbol.
            side: BUY or SELL.
            quantity: Strictly positive fill quantity.
            price: Strictly positive fill price.
            commission: Optional commission charge (always reduces cash).

        Returns:
            Realised PnL increment from this fill (after commission).
        """
        if quantity <= 0:
            raise ValueError("quantity must be > 0")
        if price <= 0:
            raise ValueError("price must be > 0")
        if commission < 0:
            raise ValueError("commission must be >= 0")

        pos = self.get_position(symbol)
        realised_gross = pos.apply_fill(side, quantity, price)

        # cash side: buying decreases cash, selling increases it
        self.cash -= side.signed * quantity * price
        # commission always reduces cash and is bookkept separately
        self.cash -= commission
        self.fees_paid += commission
        return realised_gross - commission

    def mark_to_market(self, ts: datetime, marks: Mapping[str, float]) -> float:
        """Snapshot total equity and append it to ``equity_history``.

        Args:
            ts: Timestamp for this snapshot.
            marks: ``{symbol: mark_price}`` for all currently-held positions.
                Missing symbols are treated as having zero contribution.

        Returns:
            Computed equity.
        """
        eq = self.equity(marks)
        self.equity_history.append((ts, eq))
        return eq
