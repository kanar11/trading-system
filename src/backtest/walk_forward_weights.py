"""Walk-forward evaluation for weight-based allocation strategies.

:mod:`src.validation.walk_forward` rolls *signal* strategies out of
sample; this is its counterpart for the weight-frame world: any function
that turns a price history into portfolio weights (the optimisers in
:mod:`src.portfolio`, Black-Litterman posteriors, hand-made rules) is
refit on a rolling ``train_window`` and its weights held over the next
``test_window`` bars, so every traded bar is strictly out of sample.
The stitched weight path then runs through
:func:`src.backtest.weights.backtest_weights` with the usual
next-bar/turnover-cost conventions.

Weights are decided on the close of the last training bar and take
effect from the following bar (the engine's ``shift(1)``), so no fold
ever trades on information from its own test period.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.backtest.weights import backtest_weights


@dataclass
class WalkForwardWeightsResult:
    """Output of a weight-frame walk-forward run.

    Attributes:
        results: The :func:`backtest_weights` frame over the stitched
            out-of-sample weight path (gross/net returns, turnover, cost,
            equity curve).
        weights: The stitched target-weight frame actually backtested
            (all-zero before the first fold's decision bar).
        n_folds: Number of refits performed.
    """

    results: pd.DataFrame
    weights: pd.DataFrame
    n_folds: int


def walk_forward_weights(
    prices: pd.DataFrame,
    weight_fn: Callable[[pd.DataFrame], pd.Series],
    train_window: int = 252,
    test_window: int = 21,
    cost_bps: float = 10.0,
) -> WalkForwardWeightsResult:
    """Refit an allocation rule on rolling windows and trade it out of sample.

    Args:
        prices: Wide close-price frame (columns = assets), positive and
            NaN-free.
        weight_fn: Callable receiving the trailing ``train_window`` bars of
            prices and returning target weights (Series indexed by a subset
            of the asset columns; missing assets are held at 0).
        train_window: Bars of history handed to ``weight_fn`` (>= 2).
        test_window: Bars each fit is held before refitting (>= 1).
        cost_bps: Turnover cost passed to :func:`backtest_weights`.

    Returns:
        A :class:`WalkForwardWeightsResult`.

    Raises:
        ValueError: If the windows are out of range, there is not enough
            history for one fold, or ``weight_fn`` returns unknown assets
            or non-finite weights.
    """
    if train_window < 2:
        raise ValueError(f"train_window must be >= 2, got {train_window}.")
    if test_window < 1:
        raise ValueError(f"test_window must be >= 1, got {test_window}.")
    if len(prices) <= train_window:
        raise ValueError(f"need more than train_window={train_window} bars, got {len(prices)}.")

    columns = list(prices.columns)
    weights = pd.DataFrame(0.0, index=prices.index, columns=columns)

    n_folds = 0
    for start in range(train_window, len(prices), test_window):
        train = prices.iloc[start - train_window : start]
        fold_weights = weight_fn(train)

        unknown = [a for a in fold_weights.index if a not in columns]
        if unknown:
            raise ValueError(f"weight_fn returned unknown assets {unknown}.")
        aligned = fold_weights.reindex(columns, fill_value=0.0).to_numpy(dtype=float)
        if not np.isfinite(aligned).all():
            raise ValueError("weight_fn returned non-finite weights.")

        # decided on the close of the last train bar (start - 1); held via the
        # engine's shift(1) over the test bars [start, start + test_window)
        decision_rows = slice(start - 1, min(start - 1 + test_window, len(prices)))
        weights.iloc[decision_rows] = aligned
        n_folds += 1

    results = backtest_weights(prices, weights, cost_bps=cost_bps)
    return WalkForwardWeightsResult(results=results, weights=weights, n_folds=n_folds)
