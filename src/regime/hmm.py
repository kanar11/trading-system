"""Gaussian hidden Markov model for market-regime detection.

A pure-numpy univariate Gaussian HMM trained with the Baum-Welch (EM)
algorithm and decoded with Viterbi. It complements the indicator-based
detector in :mod:`src.regime.detector` with a probabilistic, data-driven
view of latent market regimes — e.g. a calm low-volatility state versus a
turbulent high-volatility state, learned directly from the return series.

No scipy: the forward-backward pass uses per-step scaling for numerical
stability and Viterbi runs in log-space. Fitting is **deterministic** —
states are initialised from data quantiles and, after training, relabelled
in ascending order of their mean, so the output is reproducible and free of
label-switching across runs.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_LOG_2PI = float(np.log(2.0 * np.pi))
_TINY = 1e-300  # smallest positive scale to avoid divide-by-zero in forward pass


@dataclass
class HMMConfig:
    """Configuration for the Gaussian HMM.

    Attributes:
        n_states: Number of latent regimes to fit (>= 1).
        max_iter: Maximum Baum-Welch iterations.
        tol: Convergence tolerance on the log-likelihood improvement.
        var_floor: Lower bound on per-state variance (guards against a state
            collapsing onto a single observation).
    """

    n_states: int = 2
    max_iter: int = 100
    tol: float = 1e-4
    var_floor: float = 1e-8


@dataclass
class HMMResult:
    """Fitted model and decoded regime path.

    Attributes:
        states: Viterbi most-likely state per observation, labelled 0..K-1 in
            ascending order of state mean (0 = lowest-mean regime).
        state_means: Per-state emission mean.
        state_vars: Per-state emission variance.
        transition: Row-stochastic state transition matrix.
        start_prob: Initial state distribution.
        posterior: Smoothed P(state | observations), shape (T, K).
        log_likelihood: Final data log-likelihood under the fitted model.
        n_iter: Number of EM iterations run.
        converged: Whether EM converged within ``tol`` before ``max_iter``.
    """

    states: np.ndarray
    state_means: np.ndarray
    state_vars: np.ndarray
    transition: np.ndarray
    start_prob: np.ndarray
    posterior: np.ndarray
    log_likelihood: float
    n_iter: int
    converged: bool


def _quantile_init(x: np.ndarray, n_states: int, var_floor: float) -> tuple[np.ndarray, np.ndarray]:
    """Initialise state means/variances from quantile buckets of the data."""
    edges = np.quantile(x, np.linspace(0.0, 1.0, n_states + 1))
    means = np.empty(n_states, dtype=float)
    variances = np.empty(n_states, dtype=float)
    full_var = float(np.var(x)) or 1.0
    for k in range(n_states):
        lo, hi = edges[k], edges[k + 1]
        # include the right edge only in the final bucket
        mask = (x >= lo) & (x <= hi) if k == n_states - 1 else (x >= lo) & (x < hi)
        bucket = x[mask]
        if bucket.size == 0:
            means[k] = float(np.mean(edges[k : k + 2]))
            variances[k] = full_var
        else:
            means[k] = float(np.mean(bucket))
            variances[k] = float(np.var(bucket))
    variances = np.maximum(variances, var_floor)
    return means, variances


def _gaussian_emissions(x: np.ndarray, means: np.ndarray, variances: np.ndarray) -> np.ndarray:
    """Emission densities B[t, k] = N(x_t | means[k], variances[k])."""
    diff2 = (x[:, None] - means[None, :]) ** 2
    coef = 1.0 / np.sqrt(2.0 * np.pi * variances[None, :])
    out: np.ndarray = coef * np.exp(-0.5 * diff2 / variances[None, :])
    return out


def _log_emissions(x: np.ndarray, means: np.ndarray, variances: np.ndarray) -> np.ndarray:
    """Log emission densities (for the log-space Viterbi decode)."""
    diff2 = (x[:, None] - means[None, :]) ** 2
    out: np.ndarray = -0.5 * (_LOG_2PI + np.log(variances[None, :])) - diff2 / (
        2.0 * variances[None, :]
    )
    return out


def _forward_backward(
    emissions: np.ndarray, start_prob: np.ndarray, transition: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Scaled forward-backward. Returns (alpha, beta, scale, log_likelihood)."""
    n_obs, n_states = emissions.shape
    alpha = np.zeros((n_obs, n_states))
    beta = np.zeros((n_obs, n_states))
    scale = np.zeros(n_obs)

    alpha[0] = start_prob * emissions[0]
    scale[0] = max(float(alpha[0].sum()), _TINY)
    alpha[0] /= scale[0]
    for t in range(1, n_obs):
        alpha[t] = (alpha[t - 1] @ transition) * emissions[t]
        scale[t] = max(float(alpha[t].sum()), _TINY)
        alpha[t] /= scale[t]

    beta[n_obs - 1] = 1.0
    for t in range(n_obs - 2, -1, -1):
        beta[t] = (transition @ (emissions[t + 1] * beta[t + 1])) / scale[t + 1]

    log_likelihood = float(np.log(scale).sum())
    return alpha, beta, scale, log_likelihood


def _viterbi(
    log_emissions: np.ndarray, start_prob: np.ndarray, transition: np.ndarray
) -> np.ndarray:
    """Most-likely state path in log-space."""
    n_obs, n_states = log_emissions.shape
    log_a = np.log(transition + _TINY)
    delta = np.zeros((n_obs, n_states))
    psi = np.zeros((n_obs, n_states), dtype=int)

    delta[0] = np.log(start_prob + _TINY) + log_emissions[0]
    for t in range(1, n_obs):
        scores = delta[t - 1][:, None] + log_a  # (prev, cur)
        psi[t] = np.argmax(scores, axis=0)
        delta[t] = scores.max(axis=0) + log_emissions[t]

    path = np.zeros(n_obs, dtype=int)
    path[n_obs - 1] = int(np.argmax(delta[n_obs - 1]))
    for t in range(n_obs - 2, -1, -1):
        path[t] = int(psi[t + 1, path[t + 1]])
    return path


def fit_gaussian_hmm(x: np.ndarray, config: HMMConfig | None = None) -> HMMResult:
    """Fit a univariate Gaussian HMM by Baum-Welch and decode with Viterbi.

    Args:
        x: 1-D array of observations (typically daily returns).
        config: Fitting configuration. Uses defaults if None.

    Returns:
        An :class:`HMMResult` with states relabelled by ascending mean.

    Raises:
        ValueError: If ``n_states`` < 1 or there are fewer than ``2 * n_states``
            observations.
    """
    cfg = config if config is not None else HMMConfig()
    obs = np.asarray(x, dtype=float).ravel()
    n_obs = obs.shape[0]
    n_states = cfg.n_states

    if n_states < 1:
        raise ValueError(f"n_states must be >= 1, got {n_states}.")
    if n_obs < 2 * n_states:
        raise ValueError(f"Need at least {2 * n_states} observations, got {n_obs}.")

    means, variances = _quantile_init(obs, n_states, cfg.var_floor)
    start_prob = np.full(n_states, 1.0 / n_states)
    if n_states == 1:
        transition = np.ones((1, 1))
    else:
        transition = np.full((n_states, n_states), 0.1 / (n_states - 1))
        np.fill_diagonal(transition, 0.9)

    prev_ll = -np.inf
    converged = False
    n_iter = 0
    while n_iter < cfg.max_iter:
        n_iter += 1
        emissions = _gaussian_emissions(obs, means, variances)
        alpha, beta, scale, log_likelihood = _forward_backward(emissions, start_prob, transition)

        gamma = alpha * beta
        gamma /= np.maximum(gamma.sum(axis=1, keepdims=True), _TINY)

        xi_sum = np.zeros((n_states, n_states))
        for t in range(n_obs - 1):
            xi_t = (
                alpha[t][:, None]
                * transition
                * (emissions[t + 1] * beta[t + 1])[None, :]
                / scale[t + 1]
            )
            xi_sum += xi_t

        # --- M-step ---
        start_prob = gamma[0]
        transition = xi_sum / np.maximum(xi_sum.sum(axis=1, keepdims=True), _TINY)
        nk = np.maximum(gamma.sum(axis=0), _TINY)
        means = (gamma * obs[:, None]).sum(axis=0) / nk
        variances = (gamma * (obs[:, None] - means[None, :]) ** 2).sum(axis=0) / nk
        variances = np.maximum(variances, cfg.var_floor)

        if abs(log_likelihood - prev_ll) < cfg.tol:
            converged = True
            break
        prev_ll = log_likelihood

    # final posterior + decode with the converged parameters
    emissions = _gaussian_emissions(obs, means, variances)
    alpha, beta, scale, final_ll = _forward_backward(emissions, start_prob, transition)
    gamma = alpha * beta
    gamma /= np.maximum(gamma.sum(axis=1, keepdims=True), _TINY)
    states = _viterbi(_log_emissions(obs, means, variances), start_prob, transition)

    # relabel states in ascending order of mean (kills label-switching)
    order: np.ndarray = np.argsort(means)
    inverse = np.empty(n_states, dtype=int)
    inverse[order] = np.arange(n_states)

    return HMMResult(
        states=inverse[states],
        state_means=means[order],
        state_vars=variances[order],
        transition=transition[order][:, order],
        start_prob=start_prob[order],
        posterior=gamma[:, order],
        log_likelihood=final_ll,
        n_iter=n_iter,
        converged=converged,
    )


def detect_hmm_regime(
    returns: pd.Series,
    n_states: int = 2,
    config: HMMConfig | None = None,
) -> pd.Series:
    """Label each observation with its most-likely latent regime.

    Convenience wrapper over :func:`fit_gaussian_hmm` that drops missing
    values and returns an integer-labelled Series aligned to the (cleaned)
    input index. With the default ``n_states=2``, label 0 is the
    lower-mean regime (typically risk-off) and label 1 the higher-mean one.

    Args:
        returns: Series of (daily) returns.
        n_states: Number of regimes. Ignored if ``config`` is given.
        config: Optional explicit HMM configuration.

    Returns:
        Series of integer regime labels named ``"hmm_state"``.
    """
    cfg = config if config is not None else HMMConfig(n_states=n_states)
    clean = pd.Series(returns).dropna()
    result = fit_gaussian_hmm(clean.to_numpy(), cfg)
    return pd.Series(result.states, index=clean.index, name="hmm_state")
