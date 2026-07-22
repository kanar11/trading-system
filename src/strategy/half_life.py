"""Ornstein-Uhlenbeck fit and mean-reversion half-life.

The half-life of a mean-reverting series answers the one question every
pairs / stat-arb trade needs before it is placed: *how long does a
dislocation take to decay?* It sets the natural z-score window, the exit
horizon, and whether the spread reverts fast enough to be worth trading
at all. It comes straight from an Ornstein-Uhlenbeck fit of the series.

The continuous OU process ``dS = θ(μ − S)dt + σ dW`` discretises (dt = 1)
to the AR(1) model ``S_t = c + φ·S_{t−1} + ε`` with ``φ = e^{−θ}`` and
``c = μ(1 − φ)``. An OLS regression of the series on its own lag recovers

    θ = −ln φ,   μ = c / (1 − φ),   half_life = ln 2 / θ = −ln 2 / ln φ.

A series with ``φ ∈ (0, 1)`` mean-reverts (finite positive half-life); a
random walk (``φ → 1``) has an infinite half-life, and an explosive or
trending series (``φ ≥ 1``) is not mean-reverting at all — both report
``half_life = inf``. Complements the cointegration screen in
:mod:`src.strategy.pairs` (run it on the fitted spread). Pure numpy.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class OUFit:
    """Ornstein-Uhlenbeck fit of a (candidate mean-reverting) series.

    Attributes:
        theta: Mean-reversion speed ``−ln φ`` (>= 0; 0 = a random walk).
        mu: Estimated long-run mean level.
        phi: AR(1) persistence ``e^{−θ}`` — the lag-1 autoregression slope.
        sigma: Standard deviation of the AR(1) residual (innovation vol).
        half_life: ``ln 2 / θ`` in bars; ``inf`` when the series does not
            mean-revert (``φ ≥ 1`` or ``φ ≤ 0``).
    """

    theta: float
    mu: float
    phi: float
    sigma: float
    half_life: float


def fit_ou(series: pd.Series) -> OUFit:
    """Fit an OU / AR(1) model to ``series`` by OLS on its own lag.

    Args:
        series: The series to fit (e.g. a pairs spread), NaN-free, at
            least 3 observations.

    Returns:
        A populated :class:`OUFit`.

    Raises:
        ValueError: If the series is too short, contains NaNs, or is
            constant (zero variance in the lagged regressor).
    """
    values = series.to_numpy(dtype=float)
    if len(values) < 3:
        raise ValueError(f"need at least 3 observations, got {len(values)}.")
    if np.isnan(values).any():
        raise ValueError("series must not contain NaNs.")

    lagged = values[:-1]
    current = values[1:]
    var = float(np.var(lagged, ddof=1))
    if var <= 0:
        raise ValueError("series is constant; cannot fit an AR(1) model.")

    # OLS slope/intercept of current on lagged
    lag_mean = float(lagged.mean())
    cur_mean = float(current.mean())
    phi = float(np.cov(current, lagged, ddof=1)[0, 1] / var)
    intercept = cur_mean - phi * lag_mean

    residuals = current - (intercept + phi * lagged)
    sigma = float(residuals.std(ddof=1)) if len(residuals) > 1 else 0.0

    if 0.0 < phi < 1.0:
        theta = -math.log(phi)
        half_life = math.log(2.0) / theta
        mu = intercept / (1.0 - phi)
    else:
        # phi >= 1 (random walk / explosive) or phi <= 0 (alternating) —
        # not a decaying mean-reverting process
        theta = max(-math.log(phi), 0.0) if phi > 0 else 0.0
        half_life = math.inf
        mu = float(values.mean())

    return OUFit(theta=theta, mu=mu, phi=phi, sigma=sigma, half_life=half_life)


def ou_half_life(series: pd.Series) -> float:
    """Mean-reversion half-life of ``series`` in bars (``inf`` if none).

    Convenience wrapper over :func:`fit_ou` returning just the half-life.

    Raises:
        ValueError: As for :func:`fit_ou`.
    """
    return fit_ou(series).half_life
