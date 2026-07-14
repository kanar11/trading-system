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


def _validated_matrix(transition_matrix: pd.DataFrame) -> np.ndarray:
    """Require a square, row-stochastic matrix and return it as ndarray."""
    matrix = transition_matrix.to_numpy(dtype=float)
    n = matrix.shape[0]
    if matrix.ndim != 2 or matrix.shape != (n, n) or n == 0:
        raise ValueError(f"transition matrix must be square and non-empty, got {matrix.shape}.")
    if list(transition_matrix.index) != list(transition_matrix.columns):
        raise ValueError("transition matrix must have identical row and column labels.")
    if (matrix < 0).any() or np.isnan(matrix).any():
        raise ValueError("transition probabilities must be non-negative and NaN-free.")
    if not np.allclose(matrix.sum(axis=1), 1.0, atol=1e-8):
        raise ValueError("every row of the transition matrix must sum to 1.")
    return matrix


def stationary_distribution(transition_matrix: pd.DataFrame) -> pd.Series:
    """Long-run regime probabilities of a Markov transition matrix.

    Solves ``π P = π`` with ``Σπ = 1`` — the left eigenvector of ``P`` for
    eigenvalue 1. This is the fraction of time the chain spends in each
    regime once transients die out; for the empirical matrix of
    :func:`regime_transition_matrix` it should approximate the observed
    label frequencies on a long sample.

    Args:
        transition_matrix: Row-stochastic P(to | from), e.g. the output of
            :func:`regime_transition_matrix` (rows must each sum to 1).

    Returns:
        Probability Series named ``"stationary"`` indexed by regime label.

    Raises:
        ValueError: If the matrix is not square/row-stochastic or its
            labels are inconsistent.
    """
    matrix = _validated_matrix(transition_matrix)
    eigenvalues, eigenvectors = np.linalg.eig(matrix.T)
    closest = int(np.argmin(np.abs(eigenvalues - 1.0)))
    pi = np.real(eigenvectors[:, closest])
    pi = np.abs(pi)
    pi = pi / pi.sum()
    return pd.Series(pi, index=transition_matrix.index, name="stationary")


def forecast_regime_probabilities(
    transition_matrix: pd.DataFrame,
    current: pd.Series | str | int,
    steps: int = 1,
) -> pd.Series:
    """Regime probabilities ``steps`` bars ahead: ``p_0 Pᵏ``.

    Args:
        transition_matrix: Row-stochastic P(to | from) with matching labels.
        current: Either a regime label (point mass on that state) or a
            probability Series over the matrix labels.
        steps: Forecast horizon in bars (>= 0; 0 returns the start
            distribution).

    Returns:
        Probability Series named ``"forecast"`` indexed by regime label.

    Raises:
        ValueError: If the matrix is invalid, ``steps`` < 0, the label is
            unknown, or a start distribution is misaligned/not a
            probability vector.
    """
    matrix = _validated_matrix(transition_matrix)
    if steps < 0:
        raise ValueError(f"steps must be >= 0, got {steps}.")

    labels = list(transition_matrix.index)
    if isinstance(current, pd.Series):
        start = current.reindex(labels).to_numpy(dtype=float)
        if np.isnan(start).any() or (start < 0).any() or not np.isclose(start.sum(), 1.0):
            raise ValueError("current must be a probability vector over the matrix labels.")
    else:
        if current not in labels:
            raise ValueError(f"unknown regime label {current!r}; expected one of {labels}.")
        start = np.zeros(len(labels))
        start[labels.index(current)] = 1.0

    forecast = start @ np.linalg.matrix_power(matrix, steps)
    return pd.Series(forecast, index=transition_matrix.index, name="forecast")


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
