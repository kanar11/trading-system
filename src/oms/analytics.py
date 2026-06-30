"""Portfolio exposure, concentration, and fill analytics.

Read-only summaries over OMS objects: portfolio exposure (gross/net/long/short,
leverage, Herfindahl concentration) from a :class:`~src.oms.portfolio.Portfolio`
and mark prices, plus a per-order fill summary (VWAP, maker/taker mix, fill
duration) over a sequence of :class:`~src.oms.order.Fill`. Inputs are never
mutated.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from src.oms.order import Fill, Liquidity
from src.oms.portfolio import Portfolio


@dataclass
class ExposureReport:
    """Snapshot of portfolio exposure at a set of mark prices.

    Attributes:
        gross_exposure: Sum of absolute position market values.
        net_exposure: Signed sum of position market values (long − short).
        long_exposure: Market value of long positions (>= 0).
        short_exposure: Absolute market value of short positions (>= 0).
        n_long: Number of long positions.
        n_short: Number of short positions.
        leverage: gross_exposure / equity (0 when equity is non-positive).
        concentration_hhi: Herfindahl index of |market value| weights in
            [0, 1]; 1 = a single position, → 0 = many equal positions.
        largest_weight: Largest single-position share of gross exposure.
    """

    gross_exposure: float
    net_exposure: float
    long_exposure: float
    short_exposure: float
    n_long: int
    n_short: int
    leverage: float
    concentration_hhi: float
    largest_weight: float


def portfolio_exposure(portfolio: Portfolio, marks: Mapping[str, float]) -> ExposureReport:
    """Compute an :class:`ExposureReport` for ``portfolio`` at ``marks``.

    Positions that are flat or whose symbol is missing from ``marks`` are
    ignored (they cannot be valued).

    Args:
        portfolio: The portfolio to analyse (not mutated).
        marks: ``{symbol: mark_price}`` for the held positions.

    Returns:
        A populated :class:`ExposureReport`.
    """
    values = [
        pos.market_value(marks[sym])
        for sym, pos in portfolio.positions.items()
        if sym in marks and not pos.is_flat
    ]

    gross = float(sum(abs(v) for v in values))
    net = float(sum(values))
    long_exposure = float(sum(v for v in values if v > 0))
    short_exposure = float(sum(-v for v in values if v < 0))
    n_long = sum(1 for v in values if v > 0)
    n_short = sum(1 for v in values if v < 0)

    equity = portfolio.equity(marks)
    leverage = gross / equity if equity > 0 else 0.0

    if gross > 0:
        weights = [abs(v) / gross for v in values]
        concentration_hhi = float(sum(w * w for w in weights))
        largest_weight = float(max(weights))
    else:
        concentration_hhi = 0.0
        largest_weight = 0.0

    return ExposureReport(
        gross_exposure=gross,
        net_exposure=net,
        long_exposure=long_exposure,
        short_exposure=short_exposure,
        n_long=n_long,
        n_short=n_short,
        leverage=leverage,
        concentration_hhi=concentration_hhi,
        largest_weight=largest_weight,
    )


@dataclass
class FillSummary:
    """Execution summary over a sequence of fills.

    Attributes:
        n_fills: Number of fills.
        total_quantity: Sum of fill quantities.
        vwap: Quantity-weighted average fill price.
        maker_fraction: Fraction of filled quantity executed as MAKER liquidity.
        duration_seconds: Wall-clock span from the first to the last fill.
    """

    n_fills: int
    total_quantity: float
    vwap: float
    maker_fraction: float
    duration_seconds: float


def summarize_fills(fills: Sequence[Fill]) -> FillSummary:
    """Summarise a sequence of fills (e.g. ``order.fills``).

    Args:
        fills: Fills to summarise.

    Returns:
        A :class:`FillSummary` (all-zero for an empty sequence).
    """
    if not fills:
        return FillSummary(0, 0.0, 0.0, 0.0, 0.0)

    total_quantity = float(sum(f.quantity for f in fills))
    notional = float(sum(f.price * f.quantity for f in fills))
    maker_quantity = float(sum(f.quantity for f in fills if f.liquidity is Liquidity.MAKER))
    timestamps = [f.ts for f in fills]
    duration = (max(timestamps) - min(timestamps)).total_seconds()

    return FillSummary(
        n_fills=len(fills),
        total_quantity=total_quantity,
        vwap=notional / total_quantity if total_quantity > 0 else 0.0,
        maker_fraction=maker_quantity / total_quantity if total_quantity > 0 else 0.0,
        duration_seconds=float(duration),
    )
