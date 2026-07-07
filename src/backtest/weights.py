"""Vectorised multi-asset backtest over a target-weight frame.

The single-asset engine (:mod:`src.backtest.engine`) consumes a ``signal``
column; rotation and allocation strategies instead emit a *wide frame of
target weights* (e.g. :func:`src.strategy.dual_momentum.dual_momentum_strategy`
or any optimiser from :mod:`src.portfolio`). This engine closes that gap:
given close prices and target weights it produces portfolio returns, turnover,
costs and an equity curve.

Conventions match the single-asset engine: weights are *decided* on a bar's
close and *held* over the next bar (``shift(1)``), and execution costs are
charged per unit of turnover. Weights are treated as maintained continuously
at their targets between decisions (intra-period drift of the weights with
returns is not modelled); rows may sum to less than 1 — the remainder is
flat cash earning nothing — and negative weights are shorts.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def backtest_weights(
    prices: pd.DataFrame,
    weights: pd.DataFrame,
    cost_bps: float = 10.0,
) -> pd.DataFrame:
    """Backtest a target-weight path over close prices.

    Args:
        prices: Wide close-price frame (columns = assets), positive and
            NaN-free.
        weights: Target weights decided at each bar's close; same index and
            columns as ``prices``, finite values (0 = flat that asset).
        cost_bps: Execution cost in basis points per unit of turnover
            (10 = 0.1% of traded notional).

    Returns:
        DataFrame indexed like ``prices`` with columns:

        * ``portfolio_return_gross`` — Σ held_weight × asset return.
        * ``turnover`` — Σ |held weight change| across assets.
        * ``cost`` — ``turnover * cost_bps / 10_000``.
        * ``portfolio_return`` — gross minus cost.
        * ``equity_curve`` — cumulative product of ``1 + portfolio_return``.

    Raises:
        ValueError: If the frames are misaligned, prices are non-positive or
            NaN, weights are non-finite, or ``cost_bps`` is negative.
    """
    if cost_bps < 0:
        raise ValueError(f"cost_bps must be >= 0, got {cost_bps}.")
    if prices.shape[1] == 0:
        raise ValueError("prices must have at least one column.")
    if not prices.index.equals(weights.index):
        raise ValueError("prices and weights must share the same index.")
    if list(prices.columns) != list(weights.columns):
        raise ValueError("prices and weights must share the same columns.")

    price_values = prices.to_numpy(dtype=float)
    if np.isnan(price_values).any() or (price_values <= 0).any():
        raise ValueError("prices must be positive and NaN-free.")
    weight_values = weights.to_numpy(dtype=float)
    if not np.isfinite(weight_values).all():
        raise ValueError("weights must be finite.")

    asset_returns = prices.pct_change().fillna(0.0).to_numpy(dtype=float)

    # decided on close t -> held over bar t+1
    held = np.vstack([np.zeros((1, weight_values.shape[1])), weight_values[:-1]])
    gross = (held * asset_returns).sum(axis=1)

    previous = np.vstack([np.zeros((1, held.shape[1])), held[:-1]])
    turnover = np.abs(held - previous).sum(axis=1)
    cost = turnover * cost_bps / 10_000.0

    net = gross - cost
    equity = np.cumprod(1.0 + net)

    return pd.DataFrame(
        {
            "portfolio_return_gross": gross,
            "turnover": turnover,
            "cost": cost,
            "portfolio_return": net,
            "equity_curve": equity,
        },
        index=prices.index,
    )
