"""Weight drift between rebalances.

The weight-frame engine (:func:`src.backtest.weights.backtest_weights`)
holds each target weight *continuously* — an idealisation it documents:
in reality nobody trades every bar to hold 60/40 exactly. Left alone, a
book drifts, because winners compound into a larger share of the
portfolio and losers shrink out of it. That drift **is** the difference
between a rebalanced and a buy-and-hold allocation, and it is what a
periodic rebalance pays turnover to undo.

:func:`drift_weights` reconstructs the actually-held weight path: each
position's value compounds at its own return, and the weights are the
value shares::

    value_i(t) = w_i(reset) · Π (1 + r_i,s),   w_i(t) = value_i(t) / Σ_j value_j(t)

With ``rebalance_every=N`` the book is snapped back to the target every
N bars (``1`` = the engine's continuous ideal, ``None`` = pure
buy-and-hold). The resulting frame drops straight back into
``backtest_weights``, so the cost of *not* rebalancing can be measured
against the cost of rebalancing.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def drift_weights(
    target_weights: pd.Series,
    returns: pd.DataFrame,
    rebalance_every: int | None = None,
) -> pd.DataFrame:
    """Weight path produced by letting positions ride between rebalances.

    Args:
        target_weights: The allocation to start from (and to snap back to
            on each rebalance); indexed by asset, must cover every column
            of ``returns``. Weights may be negative (shorts).
        returns: Per-bar asset returns (columns = assets), NaN-free.
        rebalance_every: Snap back to ``target_weights`` every N bars
            (``1`` = held exactly every bar, ``None`` = never rebalance).

    Returns:
        DataFrame of held weights aligned to ``returns`` — row ``t`` is the
        weight *after* bar ``t``'s return. Rows sum to the same total as
        ``target_weights`` (1.0 for a fully-invested book).

    Raises:
        ValueError: If assets are missing, values are non-finite,
            ``rebalance_every`` < 1, or the portfolio value goes to zero.
    """
    if rebalance_every is not None and rebalance_every < 1:
        raise ValueError(f"rebalance_every must be >= 1 when set, got {rebalance_every}.")

    assets = list(returns.columns)
    missing = [a for a in assets if a not in target_weights.index]
    if missing:
        raise ValueError(f"target_weights missing assets {missing}.")

    target: np.ndarray = target_weights.reindex(assets).to_numpy(dtype=float)
    r: np.ndarray = returns.to_numpy(dtype=float)
    if not np.isfinite(target).all() or not np.isfinite(r).all():
        raise ValueError("target_weights and returns must be finite.")

    total = float(target.sum())
    n_bars = len(returns)
    held = np.empty((n_bars, len(assets)))

    value: np.ndarray = target.copy()
    for t in range(n_bars):
        if rebalance_every is not None and t % rebalance_every == 0:
            value = target.copy()  # snap back to the target before the bar
        value = value * (1.0 + r[t])  # positions ride the bar's return
        gross = float(value.sum())
        if gross == 0.0:
            raise ValueError(f"portfolio value hit zero at bar {t}; cannot form weights.")
        held[t] = value / gross * total

    return pd.DataFrame(held, index=returns.index, columns=assets)
