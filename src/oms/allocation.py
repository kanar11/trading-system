"""Block-trade allocation across accounts (pro-rata, largest remainder).

When one parent order fills for several accounts, compliance requires a
fair, deterministic split: pro-rata to the accounts' target weights,
in whole lots, with no account drifting more than a lot from its fair
share. Naive rounding breaks the total; this module uses the
largest-remainder (Hamilton) method: every account gets the floor of its
exact share, and the leftover lots go to the largest fractional
remainders, ties broken by account name so reruns allocate identically.

Sub-lot residue of the parent quantity (``total mod lot_size``) is left
unallocated by design — it cannot be booked in whole lots; the caller
sees it as ``total - sum(allocation)``.
"""

from __future__ import annotations

import math
from collections.abc import Mapping


def pro_rata_allocation(
    total_quantity: float,
    targets: Mapping[str, float],
    lot_size: float = 1.0,
) -> dict[str, float]:
    """Split a filled quantity across accounts pro-rata in whole lots.

    Args:
        total_quantity: Parent fill quantity to distribute (>= 0).
        targets: ``{account: weight}``; weights are any non-negative
            numbers (they are normalised internally), at least one > 0.
        lot_size: Allocation granularity (> 0), e.g. 1.0 shares or 100.0
            for board lots.

    Returns:
        ``{account: quantity}`` covering every account (zero-weight
        accounts get 0.0); quantities are multiples of ``lot_size`` and
        sum to ``floor(total_quantity / lot_size) * lot_size``.

    Raises:
        ValueError: If ``total_quantity`` < 0, ``lot_size`` <= 0,
            ``targets`` is empty, or any weight is negative/non-finite
            (or all are zero).
    """
    if total_quantity < 0:
        raise ValueError(f"total_quantity must be >= 0, got {total_quantity}.")
    if lot_size <= 0:
        raise ValueError(f"lot_size must be > 0, got {lot_size}.")
    if not targets:
        raise ValueError("targets must not be empty.")
    for account, weight in targets.items():
        if not math.isfinite(weight) or weight < 0:
            raise ValueError(f"weight for {account!r} must be finite and >= 0, got {weight}.")
    weight_sum = float(sum(targets.values()))
    if weight_sum <= 0:
        raise ValueError("at least one target weight must be > 0.")

    n_lots = int(math.floor(total_quantity / lot_size + 1e-9))
    allocation = dict.fromkeys(targets, 0.0)
    if n_lots == 0:
        return allocation

    exact = {a: w / weight_sum * n_lots for a, w in targets.items()}
    base = {a: int(math.floor(share)) for a, share in exact.items()}
    leftover = n_lots - sum(base.values())

    # largest fractional remainders first; account name breaks ties so the
    # allocation is deterministic across reruns
    by_remainder = sorted(targets, key=lambda a: (-(exact[a] - base[a]), a))
    for account in by_remainder[:leftover]:
        base[account] += 1

    for account, lots in base.items():
        allocation[account] = lots * lot_size
    return allocation
