"""Realistic execution-cost model.

The default backtest engine charges a flat ``transaction_cost`` per
unit of position turnover. That's fine for a first pass, but it
systematically *under*-estimates costs at scale because it ignores:

    1. The bid-ask spread (a fixed cost per round trip).
    2. Market impact, which grows non-linearly with size — the
       standard "square-root law" says impact scales with √(size / ADV).
    3. Per-trade fixed costs (commission, exchange fees).

This module provides an opt-in execution model that the backtest
engine can use in place of the flat cost. It is intentionally a
*reduced-form* model — not a full microstructure simulator — but it
captures the qualitative shape of real-world execution costs.

Cost as a fraction of notional on a single trade of size ``delta``:

    spread_component = 0.5 * spread_bps / 10_000
    impact_component = impact_coeff * (|delta| / participation_cap) ** impact_exponent
    fixed_component  = fixed_cost_per_trade / |notional|
    total            = spread_component + impact_component + fixed_component

``delta`` is the change in position size as a fraction of equity
(matching how the engine tracks turnover).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class ExecutionConfig:
    """Configuration for realistic execution costs.

    Attributes:
        spread_bps: Round-trip bid-ask spread in basis points
            (e.g. 5 = 0.05%). Half is charged per trade direction.
        impact_coeff: Multiplier on the market-impact term. Typical
            empirical estimates land in the 0.05 – 0.5 range depending
            on the instrument.
        impact_exponent: Power on the size term (0.5 = square-root law,
            1.0 = linear, 0 = constant).
        participation_cap: Reference size at which the impact term
            equals ``impact_coeff``. Treat the size term as
            ``(|delta| / participation_cap) ** exponent``.
        fixed_cost_per_trade: Flat per-trade cost as a fraction of
            notional (e.g. 0.00005 = 0.5 bps commission).
        min_trade_size: Trades with absolute size below this are
            considered zero-cost (filters out floating-point noise).
    """

    spread_bps: float = 5.0
    impact_coeff: float = 0.10
    impact_exponent: float = 0.5
    participation_cap: float = 1.0
    fixed_cost_per_trade: float = 0.0
    min_trade_size: float = 1e-9


def compute_execution_cost(
    trade_size: float | np.ndarray,
    config: ExecutionConfig | None = None,
) -> float | np.ndarray:
    """Return the execution cost as a *fraction of notional*.

    Scalar or vectorised — works element-wise on numpy arrays / pandas
    Series so it can be applied directly to the engine's ``trade``
    column.

    Args:
        trade_size: Absolute change in position (turnover).
        config: Execution configuration. Uses defaults if None.

    Returns:
        Cost as a fraction of notional traded (per unit). Always >= 0.
    """
    config = config or ExecutionConfig()

    size = np.abs(np.asarray(trade_size, dtype=float))
    active = size > config.min_trade_size

    spread = 0.5 * config.spread_bps / 10_000.0

    # impact = coeff * (size / cap) ** exponent — vectorised
    with np.errstate(invalid="ignore", divide="ignore"):
        scaled = np.where(
            config.participation_cap > 0,
            size / config.participation_cap,
            0.0,
        )
        impact = config.impact_coeff * np.power(scaled, config.impact_exponent)
        impact = np.where(active, impact, 0.0)

    # fixed cost is per trade, expressed already as a fraction
    fixed = np.where(active, config.fixed_cost_per_trade, 0.0)

    cost: np.ndarray = (spread + fixed) * active.astype(float) + impact

    if np.isscalar(trade_size):
        return float(cost)
    return cost


def apply_execution_costs(
    df: pd.DataFrame,
    config: ExecutionConfig | None = None,
    trade_col: str = "trade",
) -> pd.DataFrame:
    """Replace the engine's flat ``transaction_cost`` column with the
    realistic execution model.

    Mutates a copy of ``df`` and returns it. Expects ``df`` to already
    have a turnover column (default ``"trade"``) produced by the
    backtest engine.

    Unit conversion: :func:`compute_execution_cost` returns the cost per
    unit of *notional traded*, while ``strategy_returns`` are expressed
    as a fraction of *equity*. The per-unit cost is therefore multiplied
    by the bar's turnover (notional traded as a fraction of equity) —
    exactly how the engine applies its flat ``transaction_cost``. A tiny
    rebalance pays proportionally less than a full position flip.

    Args:
        df: Backtest output DataFrame.
        config: Execution configuration.
        trade_col: Column with absolute per-bar position changes.

    Returns:
        DataFrame with replaced ``transaction_cost`` and recomputed
        ``strategy_returns`` / ``equity_curve``.
    """
    if trade_col not in df.columns:
        raise ValueError(f"DataFrame must contain a '{trade_col}' column.")
    if "strategy_returns_gross" not in df.columns:
        raise ValueError("DataFrame must contain 'strategy_returns_gross'.")

    df = df.copy()
    turnover = np.abs(df[trade_col].to_numpy(dtype=float))
    per_unit_cost = np.asarray(compute_execution_cost(turnover, config), dtype=float)
    df["transaction_cost"] = per_unit_cost * turnover
    df["strategy_returns"] = df["strategy_returns_gross"] - df["transaction_cost"]
    df["equity_curve"] = (1 + df["strategy_returns"]).cumprod()
    return df
