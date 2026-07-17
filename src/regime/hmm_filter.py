"""Causal (forward-filtered) HMM regime probabilities.

The smoothed posterior in :class:`~src.regime.hmm.HMMResult` conditions
on the *whole* sample — great for historical analysis, unusable for
trading decisions, which is exactly the look-ahead caveat documented on
:func:`~src.strategy.hmm_regime.hmm_regime_strategy` since it shipped.
This module closes it: given a fitted model, the scaled forward
recursion produces ``P(state_t | observations up to t)`` — strictly
causal, so the probabilities at bar ``t`` can size a trade at bar
``t+1``. Fit the model on a training window (or walk-forward), then
filter new observations as they arrive.

The parameters are taken as fixed (no re-estimation inside the filter);
combine with :func:`src.regime.hmm.fit_gaussian_hmm` on a rolling window
for a fully out-of-sample regime pipeline.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.regime.hmm import _TINY, HMMResult, _gaussian_emissions


def filter_hmm_probabilities(
    observations: pd.Series,
    model: HMMResult,
) -> pd.DataFrame:
    """Causal state probabilities under a fitted Gaussian HMM.

    Runs the scaled forward recursion with the model's fitted parameters:
    row ``t`` is ``P(state_t | x_1..x_t)`` — it never looks past ``t``
    (verified property: truncating the future leaves earlier rows
    bit-identical).

    Args:
        observations: New observation series (e.g. returns), NaN-free.
        model: A fitted :class:`HMMResult` (means, variances, transition,
            start distribution are used; the decoded path is not).

    Returns:
        DataFrame aligned to ``observations`` with one column per state
        (0..K-1, the model's mean-sorted labels); rows sum to 1.

    Raises:
        ValueError: If ``observations`` is empty or contains NaNs.
    """
    x = observations.to_numpy(dtype=float)
    if len(x) == 0:
        raise ValueError("observations must not be empty.")
    if np.isnan(x).any():
        raise ValueError("observations must not contain NaNs.")

    emissions = _gaussian_emissions(x, model.state_means, model.state_vars)
    n_obs, n_states = emissions.shape

    filtered = np.zeros((n_obs, n_states))
    step = model.start_prob * emissions[0]
    filtered[0] = step / max(float(step.sum()), _TINY)
    for t in range(1, n_obs):
        step = (filtered[t - 1] @ model.transition) * emissions[t]
        filtered[t] = step / max(float(step.sum()), _TINY)

    return pd.DataFrame(filtered, index=observations.index, columns=list(range(n_states)))


def filtered_hmm_states(observations: pd.Series, model: HMMResult) -> pd.Series:
    """Most likely state per bar under the causal filter (argmax of
    :func:`filter_hmm_probabilities`).

    Returns:
        Integer Series named ``"state"`` aligned to ``observations``.
    """
    probabilities = filter_hmm_probabilities(observations, model)
    states = probabilities.to_numpy().argmax(axis=1)
    return pd.Series(states, index=observations.index, name="state")
