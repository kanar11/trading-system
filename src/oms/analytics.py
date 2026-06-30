"""Portfolio exposure and concentration analytics.

Read-only summaries computed from a :class:`~src.oms.portfolio.Portfolio` and a
set of mark prices: gross/net/long/short exposure, leverage, and a Herfindahl
concentration index. Useful for a risk dashboard or a post-bar snapshot. The
portfolio is never mutated.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

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
