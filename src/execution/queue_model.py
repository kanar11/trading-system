"""FIFO queue-position model for resting limit orders.

:func:`src.execution.fills.simulate_limit_fills` answers *whether* a
touched limit could fill — the optimistic bound. In a price-time-priority
market the realistic question is *how much of the queue in front trades
first*: a resting order fills only after the ``queue_ahead`` volume at
its price level has been consumed. This module walks that FIFO queue bar
by bar: each bar's volume traded *at the limit price* first erodes the
queue ahead, and only the remainder fills the order.

The two models bracket reality: optimistic touch from ``fills`` above,
FIFO queue burn-through here — a fill assumption that survives both is
robust to queue position.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def simulate_queue_fill(
    traded_volume: pd.Series,
    order_quantity: float,
    queue_ahead: float = 0.0,
) -> pd.DataFrame:
    """Walk a resting order through a FIFO queue, bar by bar.

    Args:
        traded_volume: Volume traded *at the order's price level* per bar
            (>= 0, NaN-free) — e.g. total bar volume scaled by an assumed
            at-level participation.
        order_quantity: Resting order size (> 0).
        queue_ahead: Volume queued ahead at arrival (>= 0).

    Returns:
        DataFrame aligned to ``traded_volume`` with columns:

        * ``queue_remaining`` — volume still ahead after the bar.
        * ``filled`` — quantity filled on this bar.
        * ``cumulative_filled`` — running fill total.
        * ``complete`` — True once the order is fully filled.

    Raises:
        ValueError: If ``order_quantity`` <= 0, ``queue_ahead`` < 0, or
            the volume series is negative/NaN.
    """
    if order_quantity <= 0:
        raise ValueError(f"order_quantity must be > 0, got {order_quantity}.")
    if queue_ahead < 0:
        raise ValueError(f"queue_ahead must be >= 0, got {queue_ahead}.")
    volume = traded_volume.to_numpy(dtype=float)
    if np.isnan(volume).any() or (volume < 0).any():
        raise ValueError("traded_volume must be non-negative and NaN-free.")

    n = len(volume)
    queue_remaining = np.zeros(n)
    filled = np.zeros(n)
    cumulative = np.zeros(n)

    ahead = float(queue_ahead)
    total_filled = 0.0
    for i in range(n):
        available = volume[i]
        eaten = min(ahead, available)
        ahead -= eaten
        fill = min(available - eaten, order_quantity - total_filled)
        total_filled += fill
        queue_remaining[i] = ahead
        filled[i] = fill
        cumulative[i] = total_filled

    return pd.DataFrame(
        {
            "queue_remaining": queue_remaining,
            "filled": filled,
            "cumulative_filled": cumulative,
            "complete": cumulative >= order_quantity - 1e-12,
        },
        index=traded_volume.index,
    )
