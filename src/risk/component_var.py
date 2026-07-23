"""Component and marginal Value-at-Risk (Euler risk decomposition).

A portfolio VaR number tells you how much you can lose; it does not tell
you *who is responsible*. The risk desk's answer is the Euler
decomposition of Gaussian VaR by asset:

    σ_p  = sqrt(wᵀΣw),                VaR_p = z · σ_p
    mVaR_i = ∂VaR_p/∂w_i = z · (Σw)_i / σ_p        (marginal)
    cVaR_i = w_i · mVaR_i                          (component)

Because VaR is homogeneous of degree one in the weights, Euler's theorem
makes the components **add up exactly** to the portfolio VaR — so
``component_var`` is a true budget: each line is the risk that position
actually consumes, and a hedge shows up as a negative contribution.
Marginal VaR is the per-unit sensitivity, i.e. what one more unit of
weight would add — the number that decides where to trim.

This decomposes by **asset**; :func:`src.risk.factor_var.factor_model_var`
decomposes the same risk by **factor**, and
:func:`src.portfolio.analytics.risk_contributions` reports the same split
in fractional *variance* units rather than VaR units.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _aligned_inputs(weights: pd.Series, cov: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Validate and align the weight vector and covariance matrix."""
    assets = list(weights.index)
    missing = [a for a in assets if a not in cov.index or a not in cov.columns]
    if missing:
        raise ValueError(f"cov missing assets {missing}.")

    w: np.ndarray = weights.to_numpy(dtype=float)
    sigma: np.ndarray = cov.loc[assets, assets].to_numpy(dtype=float)
    if not np.isfinite(w).all() or not np.isfinite(sigma).all():
        raise ValueError("weights and cov must be finite.")
    return w, sigma


def _quantile(level: float) -> float:
    """Positive-loss Gaussian multiplier at ``level`` (0.05 -> ~1.645)."""
    if not 0.0 < level < 1.0:
        raise ValueError(f"level must be in (0, 1), got {level}.")
    from src.validation.stat_tests import _norm_quantile

    return float(-_norm_quantile(level))


def marginal_var(weights: pd.Series, cov: pd.DataFrame, level: float = 0.05) -> pd.Series:
    """Per-asset marginal VaR ``∂VaR_p/∂w_i`` (sensitivity per unit weight).

    Args:
        weights: Portfolio weights per asset (shorts negative).
        cov: Covariance matrix covering every asset in ``weights`` (extra
            assets and label order are ignored).
        level: Tail probability (0.05 = 95% VaR).

    Returns:
        Series named ``"marginal_var"`` indexed like ``weights``; all-zero
        when the portfolio volatility is zero.

    Raises:
        ValueError: If assets are missing from ``cov``, values are not
            finite, or ``level`` is outside (0, 1).
    """
    z = _quantile(level)
    w, sigma = _aligned_inputs(weights, cov)

    sigma_w: np.ndarray = sigma @ w
    portfolio_vol = float(np.sqrt(max(float(w @ sigma_w), 0.0)))
    if portfolio_vol <= 0:
        return pd.Series(np.zeros(len(w)), index=weights.index, name="marginal_var")

    marginal: np.ndarray = z * sigma_w / portfolio_vol
    return pd.Series(marginal, index=weights.index, name="marginal_var")


def component_var(weights: pd.Series, cov: pd.DataFrame, level: float = 0.05) -> pd.Series:
    """Per-asset component VaR ``w_i · ∂VaR_p/∂w_i`` (a true risk budget).

    By Euler's theorem the components sum **exactly** to the portfolio
    parametric VaR ``z · sqrt(wᵀΣw)``. A position that hedges the book
    carries a negative component.

    Args:
        weights: Portfolio weights per asset (shorts negative).
        cov: Covariance matrix covering every asset in ``weights``.
        level: Tail probability (0.05 = 95% VaR).

    Returns:
        Series named ``"component_var"`` indexed like ``weights``.

    Raises:
        ValueError: As for :func:`marginal_var`.
    """
    marginal = marginal_var(weights, cov, level=level)
    return (weights * marginal).rename("component_var")
