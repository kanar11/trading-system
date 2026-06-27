"""Covariance-aware portfolio weight optimisation.

Extends ``src.portfolio.portfolio`` (which only knows equal / inverse-vol
weights) with three correlation-aware allocation schemes:

    * ``min_variance_weights``     — closed-form, long-only by default.
    * ``max_sharpe_weights``       — closed-form tangency portfolio.
    * ``risk_parity_weights``      — iterative equal risk-contribution.

All routines accept either a returns DataFrame (covariance is estimated
internally with a small ridge for numerical stability) or a pre-computed
covariance matrix. They never short — negative weights from the
closed-form solutions are clipped and the result re-normalised. For
strictly long-only optimal solutions you'd want a QP solver, but in
practice this clip-and-renormalise is a good approximation for
typical multi-asset baskets.

No scipy required — pure numpy.
"""

from __future__ import annotations

import logging
import math

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _covariance_matrix(
    returns: pd.DataFrame,
    cov: pd.DataFrame | np.ndarray | None,
    ridge: float = 1e-8,
) -> np.ndarray:
    """Resolve ``cov`` (caller-supplied or estimated from ``returns``) and
    return it as a numpy array with a small diagonal ridge for stability."""
    if cov is None:
        cov_arr = returns.cov().values
    elif isinstance(cov, pd.DataFrame):
        cov_arr = cov.values
    else:
        cov_arr = np.asarray(cov, dtype=float)

    n_cov = cov_arr.shape[0]
    if cov_arr.ndim != 2 or cov_arr.shape != (n_cov, n_cov):
        raise ValueError(f"covariance must be square, got shape {cov_arr.shape}")
    if cov is not None and n_cov != returns.shape[1]:
        raise ValueError(
            f"covariance shape {cov_arr.shape} does not match "
            f"returns with {returns.shape[1]} columns"
        )
    return cov_arr + ridge * np.eye(n_cov)


def _clip_and_normalise(weights: np.ndarray) -> np.ndarray:
    """Clip negative weights to zero and re-normalise to sum to 1.

    Falls back to equal weights if the entire vector is non-positive.
    """
    clipped: np.ndarray = np.maximum(weights, 0.0)
    total = float(clipped.sum())
    if total <= 0:
        return np.full_like(weights, 1.0 / len(weights))
    return clipped / total


def min_variance_weights(
    returns: pd.DataFrame,
    cov: pd.DataFrame | np.ndarray | None = None,
) -> pd.Series:
    """Long-only minimum-variance portfolio weights.

    Closed form (unconstrained):
        w = Σ⁻¹ 1 / (1ᵀ Σ⁻¹ 1)

    Negative weights are clipped to zero and the vector is re-normalised.

    Args:
        returns: DataFrame of asset returns (columns = tickers).
        cov: Optional pre-computed covariance matrix.

    Returns:
        Series of weights indexed by ticker, summing to 1.
    """
    Σ = _covariance_matrix(returns, cov)
    ones = np.ones(Σ.shape[0])
    try:
        inv = np.linalg.inv(Σ)
    except np.linalg.LinAlgError:
        logger.warning("covariance not invertible; falling back to equal weights")
        n = Σ.shape[0]
        return pd.Series(np.full(n, 1.0 / n), index=returns.columns)

    raw = inv @ ones
    w = _clip_and_normalise(raw / (ones @ raw))
    return pd.Series(w, index=returns.columns, name="min_variance")


def max_sharpe_weights(
    returns: pd.DataFrame,
    cov: pd.DataFrame | np.ndarray | None = None,
    rf_daily: float = 0.0,
) -> pd.Series:
    """Long-only tangency (max Sharpe) portfolio weights.

    Closed form (unconstrained):
        w = Σ⁻¹ (μ - rf) / (1ᵀ Σ⁻¹ (μ - rf))

    Negative weights are clipped to zero and the vector is re-normalised.

    Args:
        returns: DataFrame of asset returns (columns = tickers).
        cov: Optional pre-computed covariance matrix.
        rf_daily: Daily risk-free rate.

    Returns:
        Series of weights indexed by ticker, summing to 1.
    """
    Σ = _covariance_matrix(returns, cov)
    excess = returns.mean().to_numpy() - rf_daily

    try:
        inv = np.linalg.inv(Σ)
    except np.linalg.LinAlgError:
        logger.warning("covariance not invertible; falling back to equal weights")
        n = Σ.shape[0]
        return pd.Series(np.full(n, 1.0 / n), index=returns.columns)

    raw = inv @ excess
    denom = float(np.ones(Σ.shape[0]) @ raw)
    if denom == 0 or not np.isfinite(denom):
        # zero net excess return — fall back to min-variance
        return min_variance_weights(returns, cov).rename("max_sharpe")

    w = _clip_and_normalise(raw / denom)
    return pd.Series(w, index=returns.columns, name="max_sharpe")


def risk_parity_weights(
    returns: pd.DataFrame,
    cov: pd.DataFrame | np.ndarray | None = None,
    max_iter: int = 500,
    tol: float = 1e-8,
) -> pd.Series:
    """Equal risk-contribution (risk-parity) portfolio weights.

    Solves for w such that each asset contributes the same amount to
    total portfolio variance: w_i * (Σw)_i = constant for all i.

    Uses the cyclical coordinate-descent algorithm of Maillard,
    Roncalli & Teïletche (2010): at each step, update one weight
    analytically while holding the others fixed by solving the
    scalar quadratic

        Σ_ii · w_i² + m_i · w_i − b_i = 0,
        m_i = Σ_{j≠i} Σ_ij · w_j,

    which is the partial KKT condition for the unconstrained
    log-barrier objective ½ wᵀ Σ w − Σ_i b_i log w_i. This update
    is known to converge globally for positive-definite Σ. Weights
    are renormalised to sum to 1 at the end.

    Args:
        returns: DataFrame of asset returns.
        cov: Optional pre-computed covariance matrix.
        max_iter: Iteration cap.
        tol: Convergence tolerance on the weight L1 change.

    Returns:
        Series of weights indexed by ticker, summing to 1.
    """
    Σ = _covariance_matrix(returns, cov)
    n = Σ.shape[0]
    b = np.full(n, 1.0 / n)  # equal risk budget per asset
    w = np.full(n, 1.0 / n)

    for _ in range(max_iter):
        w_prev = w.copy()
        for i in range(n):
            a = float(Σ[i, i])
            m = float(Σ[i] @ w) - a * w[i]
            # solve a w_i² + m w_i - b_i = 0, positive root
            disc = m * m + 4 * a * b[i]
            w[i] = (-m + float(np.sqrt(disc))) / (2 * a)
        if np.abs(w - w_prev).sum() < tol:
            break

    w = w / w.sum()
    return pd.Series(w, index=returns.columns, name="risk_parity")


def maximum_diversification_weights(
    returns: pd.DataFrame,
    cov: pd.DataFrame | np.ndarray | None = None,
) -> pd.Series:
    """Long-only most-diversified portfolio (Choueifaty & Coignard, 2008).

    Maximises the diversification ratio ``(wᵀσ) / sqrt(wᵀΣw)``; the
    unconstrained solution is proportional to ``Σ⁻¹ σ`` (σ = per-asset
    volatilities). Negative weights are clipped and the vector re-normalised.
    For uncorrelated assets this reduces to inverse-volatility weighting.

    Args:
        returns: DataFrame of asset returns (columns = tickers).
        cov: Optional pre-computed covariance matrix.

    Returns:
        Series of weights indexed by ticker, summing to 1.
    """
    cov_matrix = _covariance_matrix(returns, cov)
    vols = np.sqrt(np.diag(cov_matrix))
    try:
        inv = np.linalg.inv(cov_matrix)
    except np.linalg.LinAlgError:
        logger.warning("covariance not invertible; falling back to equal weights")
        n = cov_matrix.shape[0]
        return pd.Series(np.full(n, 1.0 / n), index=returns.columns, name="max_diversification")

    raw: np.ndarray = inv @ vols
    w = _clip_and_normalise(raw)
    return pd.Series(w, index=returns.columns, name="max_diversification")


def _cluster_distance(dist: np.ndarray, left: list[int], right: list[int]) -> float:
    """Single-linkage distance between two clusters (min pairwise)."""
    return float(min(dist[i, j] for i in left for j in right))


def _single_linkage_order(dist: np.ndarray) -> list[int]:
    """Agglomerative single-linkage leaf order (quasi-diagonalisation).

    Repeatedly merges the two closest clusters and concatenates their leaf
    orders, so correlated assets end up adjacent in the returned ordering.
    """
    n = dist.shape[0]
    clusters: dict[int, list[int]] = {i: [i] for i in range(n)}
    active = list(range(n))
    next_id = n
    while len(active) > 1:
        best = math.inf
        pair = (active[0], active[1])
        for ii in range(len(active)):
            for jj in range(ii + 1, len(active)):
                a, b = active[ii], active[jj]
                d = _cluster_distance(dist, clusters[a], clusters[b])
                if d < best:
                    best = d
                    pair = (a, b)
        a, b = pair
        clusters[next_id] = clusters[a] + clusters[b]
        active.remove(a)
        active.remove(b)
        active.append(next_id)
        del clusters[a], clusters[b]
        next_id += 1
    return clusters[active[0]]


def _cluster_variance(cov: np.ndarray, items: list[int]) -> float:
    """Inverse-variance-weighted variance of a sub-portfolio."""
    sub = cov[np.ix_(items, items)]
    ivp = 1.0 / np.diag(sub)
    ivp = ivp / ivp.sum()
    return float(ivp @ sub @ ivp)


def _recursive_bisection(cov: np.ndarray, order: list[int]) -> np.ndarray:
    """Allocate weights down the dendrogram by inverse-cluster-variance splits."""
    w = np.ones(cov.shape[0])
    clusters: list[list[int]] = [order]
    while clusters:
        nxt: list[list[int]] = []
        for cluster in clusters:
            if len(cluster) <= 1:
                continue
            mid = len(cluster) // 2
            left, right = cluster[:mid], cluster[mid:]
            var_left = _cluster_variance(cov, left)
            var_right = _cluster_variance(cov, right)
            alpha = 1.0 - var_left / (var_left + var_right)
            for i in left:
                w[i] *= alpha
            for i in right:
                w[i] *= 1.0 - alpha
            nxt.append(left)
            nxt.append(right)
        clusters = nxt
    return w


def hierarchical_risk_parity_weights(
    returns: pd.DataFrame,
    cov: pd.DataFrame | np.ndarray | None = None,
) -> pd.Series:
    """Hierarchical Risk Parity weights (López de Prado, 2016).

    Three stages, all pure numpy: (1) tree clustering of assets by a
    correlation distance ``sqrt(0.5 * (1 - corr))`` via single linkage;
    (2) quasi-diagonalisation — reorder assets so similar ones are adjacent;
    (3) recursive bisection — split the ordered list and allocate between the
    halves by inverse cluster variance. Long-only by construction.

    Args:
        returns: DataFrame of asset returns (columns = tickers).
        cov: Optional pre-computed covariance matrix.

    Returns:
        Series of weights indexed by ticker, summing to 1.
    """
    cov_matrix = _covariance_matrix(returns, cov)
    n = cov_matrix.shape[0]
    if n == 1:
        return pd.Series([1.0], index=returns.columns, name="hrp")

    vols = np.sqrt(np.diag(cov_matrix))
    corr = np.clip(cov_matrix / np.outer(vols, vols), -1.0, 1.0)
    dist = np.sqrt(0.5 * (1.0 - corr))

    order = _single_linkage_order(dist)
    w = _recursive_bisection(cov_matrix, order)
    return pd.Series(w, index=returns.columns, name="hrp")
