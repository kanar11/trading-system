"""Combine signals from multiple strategies into one.

The single-strategy backtest pipeline outputs a per-day signal in
{-1, 0, +1}. When several uncorrelated signals are available a
common trick is to combine them — it tends to reduce per-strategy
noise and avoids the model-selection problem of "which one do I run
on this data?".

Three combiners are provided. All operate row-wise on a DataFrame
whose columns are individual signal series. They return a single
``pandas.Series`` ready to slot into the backtest engine.

    * ``majority_vote``    — sign of the row sum, breaks ties to 0.
    * ``weighted_sum``     — weighted average, sign-thresholded.
    * ``unanimous``        — all strategies must agree; otherwise flat.
"""

from __future__ import annotations

import logging
from typing import Iterable, Mapping

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _validate_signal_matrix(signals: pd.DataFrame) -> None:
    if signals.empty:
        raise ValueError("signals frame is empty")
    bad = ~signals.isin([-1, 0, 1]).all(axis=0)
    bad_cols = bad[bad].index.tolist()
    if bad_cols:
        raise ValueError(f"Signal columns must be in {{-1, 0, 1}}: {bad_cols}")


def majority_vote(signals: pd.DataFrame) -> pd.Series:
    """Sign of the row sum.

    For an even number of signals a tie (row sum == 0) returns 0
    (flat). This is the cheapest combiner and the most robust to
    a single misbehaving strategy.

    Args:
        signals: DataFrame whose columns are individual signal series
            taking values in {-1, 0, 1}.

    Returns:
        Combined signal series in {-1, 0, 1}.
    """
    _validate_signal_matrix(signals)
    row_sum = signals.sum(axis=1)
    combined = np.sign(row_sum).astype(int)
    return pd.Series(combined.values, index=signals.index, name="ensemble")


def weighted_sum(
    signals: pd.DataFrame,
    weights: Mapping[str, float] | Iterable[float] | None = None,
    threshold: float = 0.0,
) -> pd.Series:
    """Sign-thresholded weighted average of signals.

    Useful when one strategy has higher confidence (e.g. fitted on a
    larger sample, or known to be regime-appropriate today).

    Args:
        signals: DataFrame whose columns are individual signal series.
        weights: Mapping {column → weight} or an iterable of weights
            in the column order. If None, each column gets weight 1.
        threshold: Minimum absolute weighted sum to leave the flat
            zone. Setting threshold > 0 reduces signal churn.

    Returns:
        Combined signal series in {-1, 0, 1}.
    """
    _validate_signal_matrix(signals)
    cols = list(signals.columns)
    if weights is None:
        w = np.ones(len(cols))
    elif isinstance(weights, Mapping):
        w = np.array([weights.get(c, 0.0) for c in cols], dtype=float)
    else:
        w = np.asarray(list(weights), dtype=float)
        if len(w) != len(cols):
            raise ValueError(
                f"weights length {len(w)} does not match {len(cols)} signal columns"
            )

    if w.sum() == 0:
        raise ValueError("weights must not sum to zero")

    weighted = signals.values * w
    score = weighted.sum(axis=1) / w.sum()
    combined = np.where(
        score > threshold, 1, np.where(score < -threshold, -1, 0)
    ).astype(int)
    return pd.Series(combined, index=signals.index, name="ensemble")


def unanimous(signals: pd.DataFrame) -> pd.Series:
    """Take the position only if every strategy agrees.

    Most conservative combiner — a single dissenter (including a flat
    one) collapses the row to zero. Trades less often but with higher
    expected per-trade conviction.

    Args:
        signals: DataFrame whose columns are individual signal series.

    Returns:
        Combined signal series in {-1, 0, 1}.
    """
    _validate_signal_matrix(signals)
    first = signals.iloc[:, 0].values
    all_same = (signals.values == first[:, None]).all(axis=1)
    combined = np.where(all_same, first, 0).astype(int)
    return pd.Series(combined, index=signals.index, name="ensemble")
