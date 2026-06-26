"""Per-symbol position state.

Tracks the running quantity, average cost basis, and accumulated
realized PnL for a single symbol. Designed for the event-driven
engine to call ``apply_fill`` on every fill; ``Portfolio`` aggregates
across symbols.

Long and short positions are both supported. Flipping from long to
short (or vice-versa) in a single fill is handled by realising PnL
on the closing portion and reopening the remainder at the new price.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.oms.order import Side


@dataclass
class Position:
    """Single-symbol position state.

    Attributes:
        symbol: Trading symbol.
        quantity: Signed quantity. Positive = long, negative = short, 0 = flat.
        avg_price: Volume-weighted average entry price. Zero when flat.
        realized_pnl: Accumulated PnL from all closed portions of the position.
        total_traded_qty: Sum of absolute fill quantities (for turnover/commission attribution).
    """

    symbol: str
    quantity: float = 0.0
    avg_price: float = 0.0
    realized_pnl: float = 0.0
    total_traded_qty: float = 0.0

    @property
    def is_flat(self) -> bool:
        return abs(self.quantity) < 1e-12

    @property
    def is_long(self) -> bool:
        return self.quantity > 1e-12

    @property
    def is_short(self) -> bool:
        return self.quantity < -1e-12

    @property
    def cost_basis(self) -> float:
        """Notional cost of the current position (always non-negative)."""
        return abs(self.quantity) * self.avg_price

    def market_value(self, mark_price: float) -> float:
        """Mark-to-market notional value of the position."""
        return self.quantity * mark_price

    def unrealized_pnl(self, mark_price: float) -> float:
        """PnL of the open position at the supplied mark price."""
        if self.is_flat:
            return 0.0
        return (mark_price - self.avg_price) * self.quantity

    def apply_fill(self, side: Side, fill_qty: float, fill_price: float) -> float:
        """Apply a fill to this position and return the realised PnL increment.

        Handles the three cases:

        1. **Opening / adding** (same direction as current position, or flat):
           recompute volume-weighted average price.

        2. **Closing partial** (opposite direction, smaller than position):
           crystallise PnL on the closed quantity, keep avg_price for remainder.

        3. **Flipping** (opposite direction, larger than position):
           close the whole existing position (realised PnL), then open the
           remainder in the new direction at ``fill_price``.

        Args:
            side: BUY (signed +1) or SELL (signed -1).
            fill_qty: Strictly positive fill quantity.
            fill_price: Strictly positive fill price.

        Returns:
            The realised PnL added by this fill (zero for purely opening fills).
        """
        if fill_qty <= 0:
            raise ValueError(f"fill_qty must be > 0, got {fill_qty}")
        if fill_price <= 0:
            raise ValueError(f"fill_price must be > 0, got {fill_price}")

        signed_qty = side.signed * fill_qty
        realised_delta = 0.0

        if (
            self.is_flat
            or (self.quantity > 0 and signed_qty > 0)
            or (self.quantity < 0 and signed_qty < 0)
        ):
            # opening or adding in the same direction
            new_qty = self.quantity + signed_qty
            # weighted-average price by absolute quantity
            prev_notional = abs(self.quantity) * self.avg_price
            add_notional = abs(signed_qty) * fill_price
            self.avg_price = (prev_notional + add_notional) / abs(new_qty)
            self.quantity = new_qty

        else:
            # opposite direction — closing some / all / flipping
            current_dir = 1 if self.quantity > 0 else -1
            close_qty = min(fill_qty, abs(self.quantity))
            # realised PnL is (exit - entry) * signed_qty_closed
            realised_delta = (fill_price - self.avg_price) * current_dir * close_qty
            self.realized_pnl += realised_delta
            remaining_fill = fill_qty - close_qty

            new_qty = self.quantity + signed_qty
            if remaining_fill > 0 and abs(new_qty) > 1e-12:
                # flip: open the remainder at fill_price in the new direction
                self.avg_price = fill_price
            elif abs(new_qty) < 1e-12:
                # fully closed
                self.avg_price = 0.0
                new_qty = 0.0
            # else: partial close, avg_price unchanged
            self.quantity = new_qty

        self.total_traded_qty += fill_qty
        return realised_delta
