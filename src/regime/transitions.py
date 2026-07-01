"""Regime transition analytics.

Descriptive statistics over a regime-label series (from
:func:`~src.regime.detector.detect_regime`, :func:`~src.regime.hmm.detect_hmm_regime`,
or any categorical state series): the empirical first-order Markov transition
matrix and the average dwell time per regime. Pure pandas; inputs are not
mutated.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def regime_transition_matrix(regimes: pd.Series) -> pd.DataFrame:
    """Empirical first-order Markov transition probabilities.

    Args:
        regimes: Ordered series of regime labels.

    Returns:
        DataFrame of P(to | from): rows are the current regime, columns the next
        regime, each row summing to 1 (a row with no outgoing transitions is
        all-zero). Empty when there are fewer than two observations.
    """
    r = pd.Series(regimes).dropna()
    if len(r) < 2:
        return pd.DataFrame()

    labels = sorted(r.unique())
    counts = pd.crosstab(
        pd.Series(r.iloc[:-1].to_numpy(), name="from"),
        pd.Series(r.iloc[1:].to_numpy(), name="to"),
    ).reindex(index=labels, columns=labels, fill_value=0)

    row_sums = counts.sum(axis=1)
    probs = counts.div(row_sums.replace(0, np.nan), axis=0).fillna(0.0)
    return probs


def regime_durations(regimes: pd.Series) -> pd.Series:
    """Average consecutive dwell time (in bars) per regime.

    Args:
        regimes: Ordered series of regime labels.

    Returns:
        Series indexed by regime label giving the mean run length. Empty when
        there are no observations.
    """
    r = pd.Series(regimes).dropna()
    if r.empty:
        return pd.Series(dtype=float)

    block = (r != r.shift()).cumsum()
    runs = pd.DataFrame({"label": r.to_numpy(), "block": block.to_numpy()})
    per_run = runs.groupby("block").agg(label=("label", "first"), length=("label", "size"))
    out: pd.Series = per_run.groupby("label")["length"].mean()
    out.index.name = "regime"
    return out
