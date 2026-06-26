"""Abstract base class for event-driven strategies.

The vectorised strategies in this package (``momentum_strategy``,
``mean_reversion_strategy`` etc.) all return a signal series — they
operate on a whole DataFrame at once and are ideal for fast research
sweeps. The event-engine runs in a different mode: it walks the
DataFrame bar by bar and calls back into the strategy on every bar
with a live :class:`Context` (portfolio, OMS, history-to-date).

This module defines the contract for event-driven strategies and
ships a thin :class:`SmaCrossoverStrategy` that demonstrates the
full lifecycle:

    on_start          -> once, before the first bar
    on_bar(ctx)       -> per bar
    on_order_event    -> reserved for fill / cancel callbacks
    on_end            -> once, after the last bar

For simple research, just subclass :class:`Strategy` and implement
``on_bar``. The other hooks default to no-ops.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.backtest.event_engine import Context
    from src.oms import Order

logger = logging.getLogger(__name__)


class Strategy(ABC):
    """Base class for event-driven strategies.

    Lifecycle hooks (override as needed):
        * ``on_start(ctx)``  — once before the first bar.
        * ``on_bar(ctx)``    — called for every bar. **Required**.
        * ``on_order_event(order)`` — called by the engine on fills /
          cancels / rejects. Default: no-op.
        * ``on_end(ctx)``    — once after the last bar.
    """

    name: str = "Strategy"

    def on_start(self, ctx: Context) -> None:
        """Hook called once before the first bar. Default: no-op."""
        pass

    @abstractmethod
    def on_bar(self, ctx: Context) -> None:
        """Required: called once per bar.

        Implementations typically:
            1. Update indicators from ``ctx.history`` or ``ctx.bar``.
            2. Inspect ``ctx.portfolio`` to know current position.
            3. Submit orders via ``ctx.submit_order(...)``.
        """
        ...

    def on_order_event(self, order: Order) -> None:
        """Hook called by the engine after each fill / cancel / reject.

        Currently the engine does not invoke this — kept for future
        extension when the engine grows a fill-event callback.
        """
        pass

    def on_end(self, ctx: Context) -> None:
        """Hook called once after the last bar. Default: no-op."""
        pass


# ---------------------------------------------------------------------------
# Canonical example: SMA crossover
# ---------------------------------------------------------------------------


class SmaCrossoverStrategy(Strategy):
    """Classic fast/slow simple-moving-average crossover.

    Goes long ``trade_qty`` shares when the fast SMA crosses *above*
    the slow SMA; flips to short on the opposite cross. Sits flat
    until both SMAs have enough history.

    This is intentionally tiny — it exists as the reference event-driven
    strategy. Real strategies will compose indicators from
    :mod:`src.indicators` and risk controls from :mod:`src.risk`.

    Args:
        fast: Fast SMA window.
        slow: Slow SMA window (must be > fast).
        trade_qty: Position size in shares.
        allow_short: If False, short signals are clamped to flat.
    """

    name = "SmaCrossover"

    def __init__(
        self, fast: int = 10, slow: int = 30, trade_qty: float = 10, allow_short: bool = True
    ):
        if fast >= slow:
            raise ValueError(f"fast ({fast}) must be < slow ({slow})")
        self.fast = fast
        self.slow = slow
        self.trade_qty = trade_qty
        self.allow_short = allow_short
        self._prev_signal = 0

    def on_bar(self, ctx: Context) -> None:
        history = ctx.history
        if len(history) < self.slow:
            return  # warm-up

        closes = history["close"]
        fast_ma = closes.iloc[-self.fast :].mean()
        slow_ma = closes.iloc[-self.slow :].mean()

        new_signal = (
            1 if fast_ma > slow_ma else (-1 if (self.allow_short and fast_ma < slow_ma) else 0)
        )
        if new_signal == self._prev_signal:
            return

        # close any opposite position first
        pos = ctx.portfolio.get_position(ctx.symbol)
        if not pos.is_flat:
            close_side = "SELL" if pos.is_long else "BUY"
            from src.oms import Side

            ctx.submit_order(Side[close_side], abs(pos.quantity), client_tag=f"{self.name}:close")

        # then open new position in the target direction
        if new_signal != 0:
            from src.oms import Side

            side = Side.BUY if new_signal > 0 else Side.SELL
            ctx.submit_order(side, self.trade_qty, client_tag=f"{self.name}:open")

        self._prev_signal = new_signal
