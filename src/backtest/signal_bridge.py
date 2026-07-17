"""Bridge from signal strategies to the event-driven engine.

The package has sixteen signal generators that only ever ran through the
vectorised engine (frictionless fills at close-to-close returns) and an
event engine with realistic market-order fills at next-bar open,
commissions, slippage and the OMS portfolio — but nothing connecting
them. This bridge closes that gap: any DataFrame with the standard
``signal`` column replays through the event engine, so the exact same
strategy can be graded twice — research-fast and execution-realistic —
and the difference *is* the cost of trading it.

Conventions match both worlds: the signal decided on bar ``t``'s close
is submitted after that close and fills at bar ``t+1``'s open (the
engine's look-ahead-free market fill), the analogue of the vectorised
``shift(1)``. Position sizing is a fraction of current equity at the
decision close; the bridge trades only when the target *direction*
changes, so equity drift does not generate churn.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.backtest.event_engine import Context, EventEngine, EventEngineResult
from src.oms import OrderType, Side, TimeInForce


@dataclass
class SignalFollowStrategy:
    """Event-engine strategy that follows a precomputed signal series.

    On each bar the desired direction is ``sign(signal)`` (NaN = flat).
    When it differs from the held direction, one GTC market order moves
    the position to ``direction * position_fraction * equity / close``
    in a single delta trade (flips close and reopen in one order).

    Attributes:
        signals: Signal per bar, aligned to the backtest index.
        position_fraction: Fraction of equity deployed per unit of
            direction (> 0).
    """

    signals: pd.Series
    position_fraction: float = 1.0

    def on_bar(self, ctx: Context) -> None:
        raw = self.signals.get(ctx.ts, 0.0)
        direction = 0 if pd.isna(raw) else int(np.sign(raw))

        position = ctx.portfolio.positions.get(ctx.symbol)
        current_qty = position.quantity if position is not None else 0.0
        current_dir = 0 if abs(current_qty) < 1e-9 else (1 if current_qty > 0 else -1)
        if direction == current_dir:
            return

        close = float(ctx.bar["close"])
        equity = ctx.portfolio.equity({ctx.symbol: close})
        target_qty = direction * self.position_fraction * equity / close
        delta = target_qty - current_qty
        if abs(delta) < 1e-9:
            return
        ctx.submit_order(
            side=Side.BUY if delta > 0 else Side.SELL,
            quantity=abs(delta),
            order_type=OrderType.MARKET,
            tif=TimeInForce.GTC,
            client_tag="signal_bridge",
        )


def run_signal_event_backtest(
    df: pd.DataFrame,
    initial_cash: float = 100_000.0,
    position_fraction: float = 1.0,
    commission_per_share: float = 0.0,
    commission_min: float = 0.0,
    slippage_bps: float = 0.0,
    symbol: str = "ASSET",
) -> EventEngineResult:
    """Replay a signal DataFrame through the event-driven engine.

    Args:
        df: DataFrame with ``open``/``high``/``low``/``close`` and the
            package-standard ``signal`` column (e.g. any strategy
            generator's output merged with its OHLCV input).
        initial_cash: Starting cash of the OMS portfolio.
        position_fraction: Fraction of equity per unit of direction (> 0).
        commission_per_share: Engine commission per share.
        commission_min: Minimum commission per order.
        slippage_bps: Adverse slippage per fill in basis points.
        symbol: Instrument label used in the OMS.

    Returns:
        The engine's :class:`EventEngineResult` (equity curve, returns,
        final portfolio, orders and fill log).

    Raises:
        ValueError: If ``signal`` (or an OHLC column, via the engine) is
            missing, or ``position_fraction`` <= 0.
    """
    if "signal" not in df.columns:
        raise ValueError("DataFrame must contain a 'signal' column.")
    if position_fraction <= 0:
        raise ValueError(f"position_fraction must be > 0, got {position_fraction}.")

    engine = EventEngine(
        symbol=symbol,
        initial_cash=initial_cash,
        commission_per_share=commission_per_share,
        commission_min=commission_min,
        slippage_bps=slippage_bps,
    )
    follower = SignalFollowStrategy(signals=df["signal"], position_fraction=position_fraction)
    return engine.run(df, follower)
