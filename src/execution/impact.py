"""Optimal-execution scheduling and participation-based impact models.

Complements the reduced-form per-trade cost in :mod:`src.execution.slippage`
with two staples of the optimal-execution literature:

    * ``participation_rate_cost`` — temporary impact as a power law of the
      order's participation rate (size / ADV); the empirical "square-root law"
      is the ``exponent=0.5`` special case.
    * ``almgren_chriss_trajectory`` / ``almgren_chriss_cost`` — the
      Almgren-Chriss (2000) optimal liquidation schedule and its expected
      impact cost. Higher urgency front-loads trading (less timing risk, more
      impact); zero urgency is plain TWAP.

Pure numpy.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


def participation_rate_cost(
    order_size: float,
    adv: float,
    eta: float = 0.1,
    exponent: float = 1.0,
) -> float:
    """Temporary market-impact cost as a power law of participation rate.

    ``cost_fraction = eta * (|order_size| / adv) ** exponent``

    Args:
        order_size: Order quantity (shares or notional; same units as adv).
        adv: Average daily volume / liquidity reference (> 0).
        eta: Impact coefficient (the cost when participation == 1).
        exponent: Power on the participation rate (1.0 = linear, 0.5 = the
            empirical square-root law).

    Returns:
        Impact cost as a fraction of notional. 0.0 for a zero order or a
        non-positive ADV.
    """
    if adv <= 0 or order_size == 0:
        return 0.0
    participation = abs(order_size) / adv
    return float(eta * participation**exponent)


def almgren_chriss_trajectory(
    total_shares: float,
    n_steps: int,
    urgency: float = 0.0,
) -> np.ndarray:
    """Almgren-Chriss optimal liquidation trajectory (holdings over time).

    Returns the remaining position at each of ``n_steps + 1`` equally-spaced
    times, from ``total_shares`` down to 0. ``urgency`` is the trade-decay rate
    κ that balances market impact against timing (volatility) risk: κ → 0 gives
    a linear TWAP schedule, larger κ front-loads execution.

    Args:
        total_shares: Position to liquidate (sign preserved for buys).
        n_steps: Number of trading intervals (>= 1).
        urgency: Non-negative decay rate κ. 0 = TWAP.

    Returns:
        Array of length ``n_steps + 1`` of remaining holdings.

    Raises:
        ValueError: If ``n_steps`` < 1 or ``urgency`` < 0.
    """
    if n_steps < 1:
        raise ValueError(f"n_steps must be >= 1, got {n_steps}.")
    if urgency < 0:
        raise ValueError(f"urgency must be >= 0, got {urgency}.")

    k = np.arange(n_steps + 1)
    if urgency == 0:
        frac = (n_steps - k) / n_steps
    else:
        frac = np.sinh(urgency * (n_steps - k)) / np.sinh(urgency * n_steps)
    out: np.ndarray = total_shares * frac
    return out


def almgren_chriss_cost(
    trajectory: np.ndarray,
    eta: float = 0.1,
    gamma: float = 0.0,
) -> float:
    """Expected impact cost of executing a holdings ``trajectory``.

    Temporary impact is charged on each interval's traded quantity squared
    (η · Σ nₖ²); permanent impact is the linear-in-size term ½·γ·X². With a
    flat (TWAP) trajectory the temporary term reduces to η·X²/N, the minimum
    over all schedules with the same total size.

    Args:
        trajectory: Holdings over time (e.g. from
            :func:`almgren_chriss_trajectory`); ``trajectory[0]`` is the full
            position and ``trajectory[-1]`` should be ~0.
        eta: Temporary-impact coefficient.
        gamma: Permanent-impact coefficient.

    Returns:
        Total expected impact cost (same units as η · shares²).
    """
    traj = np.asarray(trajectory, dtype=float)
    if traj.size < 2:
        return 0.0
    trades = -np.diff(traj)
    temporary = float(eta * np.sum(trades**2))
    total_shares = float(traj[0])
    permanent = float(0.5 * gamma * total_shares**2)
    return temporary + permanent
