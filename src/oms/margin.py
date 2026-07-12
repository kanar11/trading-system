"""Margin requirements and buying power (Reg-T style account model).

The OMS can check orders (:mod:`src.oms.checks`) and plan rebalances
(:mod:`src.oms.rebalance`), but neither answers the broker's two
questions: *how much more can this account buy*, and *is it in a margin
call?* This module models the standard US Reg-T margin account: initial
margin (classically 50% of position notional per side) gates new trades,
maintenance margin (25% long / 30% short) triggers the call when account
equity falls below it.

Everything is derived from the live :class:`~src.oms.portfolio.Portfolio`
and mark prices; positions whose symbol has no mark are ignored, matching
the conventions of :mod:`src.oms.analytics`. Pure read-only computation —
enforcing the numbers (rejecting orders, liquidating) stays with the
caller, e.g. by wiring ``buying_power`` into a
:class:`~src.oms.checks.PreTradeLimits` policy.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from src.oms.portfolio import Portfolio


@dataclass(frozen=True)
class MarginRequirements:
    """Per-side margin rates as fractions of position notional.

    Defaults follow US Reg-T convention: 50% initial on both sides, 25%
    maintenance on longs, 30% on shorts.

    Attributes:
        initial_long: Initial margin rate on long notional.
        initial_short: Initial margin rate on short notional.
        maintenance_long: Maintenance rate on long notional.
        maintenance_short: Maintenance rate on short notional.
    """

    initial_long: float = 0.50
    initial_short: float = 0.50
    maintenance_long: float = 0.25
    maintenance_short: float = 0.30

    def __post_init__(self) -> None:
        for name in ("initial_long", "initial_short", "maintenance_long", "maintenance_short"):
            value = float(getattr(self, name))
            if not 0.0 < value <= 1.0:
                raise ValueError(f"{name} must be in (0, 1], got {value}.")
        if self.maintenance_long > self.initial_long:
            raise ValueError("maintenance_long cannot exceed initial_long.")
        if self.maintenance_short > self.initial_short:
            raise ValueError("maintenance_short cannot exceed initial_short.")


@dataclass
class MarginReport:
    """Snapshot of an account's margin state.

    Attributes:
        equity: Account equity (cash + net position value).
        long_value: Market value of long positions (>= 0).
        short_value: Absolute market value of short positions (>= 0).
        initial_margin: Initial requirement of the current book.
        maintenance_margin: Maintenance requirement of the current book.
        excess_equity: ``equity − initial_margin`` (negative = no room).
        buying_power: Additional *long* notional purchasable, i.e.
            ``max(excess_equity, 0) / initial_long`` (2x cash for a flat
            Reg-T account).
        margin_call: True when equity is below the maintenance margin.
    """

    equity: float
    long_value: float
    short_value: float
    initial_margin: float
    maintenance_margin: float
    excess_equity: float
    buying_power: float
    margin_call: bool


def margin_report(
    portfolio: Portfolio,
    marks: Mapping[str, float],
    requirements: MarginRequirements | None = None,
) -> MarginReport:
    """Compute the margin state of ``portfolio`` at ``marks``.

    Args:
        portfolio: The account to analyse (not mutated).
        marks: ``{symbol: mark_price}``; unmarked positions are ignored.
        requirements: Margin policy (defaults to Reg-T rates).

    Returns:
        A populated :class:`MarginReport`.
    """
    req = requirements if requirements is not None else MarginRequirements()

    values = [
        pos.market_value(marks[sym])
        for sym, pos in portfolio.positions.items()
        if sym in marks and not pos.is_flat
    ]
    long_value = float(sum(v for v in values if v > 0))
    short_value = float(sum(-v for v in values if v < 0))

    equity = portfolio.equity(marks)
    initial = req.initial_long * long_value + req.initial_short * short_value
    maintenance = req.maintenance_long * long_value + req.maintenance_short * short_value
    excess = equity - initial

    return MarginReport(
        equity=equity,
        long_value=long_value,
        short_value=short_value,
        initial_margin=initial,
        maintenance_margin=maintenance,
        excess_equity=excess,
        buying_power=max(excess, 0.0) / req.initial_long,
        margin_call=equity < maintenance,
    )
