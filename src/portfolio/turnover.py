"""Turnover measurement and turnover-constrained rebalancing.

Optimisers hand back target weights with no regard for how expensive it
is to get there: a fresh min-variance or Black-Litterman solution can
demand rotating half the book. The standard industry control is a
*turnover budget* — move from the current weights **toward** the targets,
but cap the one-way turnover ``Σ|w_target − w_current|`` per rebalance.

:func:`constrain_turnover` implements the proportional version: when the
desired move exceeds the budget, every position's delta is scaled by the
same factor ``λ = budget / turnover``, which preserves the *direction* of
the trade list, spends exactly the budget, and keeps the weight sum
unchanged (an affine blend of two allocations). Symbols present on only
one side are treated as weight 0 on the other, so entries and exits are
budgeted like any other trade.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _aligned(
    current_weights: pd.Series,
    target_weights: pd.Series,
) -> tuple[pd.Series, pd.Series]:
    """Align both weight vectors on the union of symbols (missing = 0)."""
    symbols = current_weights.index.union(target_weights.index)
    current = current_weights.reindex(symbols, fill_value=0.0).astype(float)
    target = target_weights.reindex(symbols, fill_value=0.0).astype(float)
    if not (np.isfinite(current.to_numpy()).all() and np.isfinite(target.to_numpy()).all()):
        raise ValueError("weights must be finite.")
    return current, target


def portfolio_turnover(current_weights: pd.Series, target_weights: pd.Series) -> float:
    """One-way turnover ``Σ|w_target − w_current|`` over the symbol union.

    Raises:
        ValueError: If any weight is NaN or infinite.
    """
    current, target = _aligned(current_weights, target_weights)
    return float((target - current).abs().sum())


def constrain_turnover(
    current_weights: pd.Series,
    target_weights: pd.Series,
    max_turnover: float,
) -> pd.Series:
    """Move toward ``target_weights`` without exceeding a turnover budget.

    If the full move fits inside ``max_turnover`` the targets are returned
    as-is; otherwise every delta is scaled by ``λ = max_turnover / turnover``
    so the constrained rebalance spends exactly the budget.

    Args:
        current_weights: Held weights per symbol.
        target_weights: Desired weights per symbol (any optimiser output).
        max_turnover: One-way turnover budget (>= 0; 0 = no trading).

    Returns:
        Weight Series named ``"weights"`` on the union of symbols; an
        affine blend of the inputs, so equal input sums are preserved.

    Raises:
        ValueError: If ``max_turnover`` < 0 or any weight is not finite.
    """
    if max_turnover < 0:
        raise ValueError(f"max_turnover must be >= 0, got {max_turnover}.")
    current, target = _aligned(current_weights, target_weights)

    turnover = float((target - current).abs().sum())
    if turnover <= max_turnover:
        return target.rename("weights")

    scale = max_turnover / turnover
    return (current + scale * (target - current)).rename("weights")
