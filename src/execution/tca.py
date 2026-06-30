"""Post-trade transaction-cost analysis (TCA).

Measures *realised* execution quality against standard benchmarks — the arrival
(decision) price and a market VWAP — complementing the pre-trade cost models in
:mod:`src.execution.slippage` / :mod:`src.execution.impact`.

Costs are returned as signed fractions of the benchmark price: positive means
the execution was worse than the benchmark (a cost), negative means price
improvement.
"""

from __future__ import annotations

import numpy as np

ArrayLike = np.ndarray | list[float]


def _sign(side: str) -> int:
    s = side.lower()
    if s == "buy":
        return 1
    if s == "sell":
        return -1
    raise ValueError(f"side must be 'buy' or 'sell', got {side!r}")


def execution_vwap(prices: ArrayLike, quantities: ArrayLike) -> float:
    """Quantity-weighted average execution price (0.0 when nothing was filled).

    Raises:
        ValueError: If ``prices`` and ``quantities`` have different lengths.
    """
    p = np.asarray(prices, dtype=float)
    q = np.abs(np.asarray(quantities, dtype=float))
    if p.shape != q.shape:
        raise ValueError(f"prices {p.shape} and quantities {q.shape} must match")
    total = float(q.sum())
    if total == 0:
        return 0.0
    return float((p * q).sum() / total)


def implementation_shortfall(
    prices: ArrayLike,
    quantities: ArrayLike,
    arrival_price: float,
    side: str = "buy",
) -> float:
    """Signed execution cost vs the arrival (decision) price, as a fraction.

    For a buy, paying above the arrival price is a positive cost; for a sell,
    receiving below it is a positive cost. Returns 0.0 when nothing was filled.

    Raises:
        ValueError: If ``arrival_price`` <= 0 or ``side`` is invalid.
    """
    if arrival_price <= 0:
        raise ValueError(f"arrival_price must be > 0, got {arrival_price}")
    sign = _sign(side)
    vwap = execution_vwap(prices, quantities)
    if vwap == 0:
        return 0.0
    return sign * (vwap - arrival_price) / arrival_price


def vwap_slippage(
    prices: ArrayLike,
    quantities: ArrayLike,
    benchmark_vwap: float,
    side: str = "buy",
) -> float:
    """Signed execution cost vs a market VWAP benchmark, as a fraction.

    Positive means the fills were worse than the market VWAP over the trading
    window. Returns 0.0 when nothing was filled.

    Raises:
        ValueError: If ``benchmark_vwap`` <= 0 or ``side`` is invalid.
    """
    if benchmark_vwap <= 0:
        raise ValueError(f"benchmark_vwap must be > 0, got {benchmark_vwap}")
    sign = _sign(side)
    vwap = execution_vwap(prices, quantities)
    if vwap == 0:
        return 0.0
    return sign * (vwap - benchmark_vwap) / benchmark_vwap
