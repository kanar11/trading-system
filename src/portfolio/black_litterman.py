"""Black-Litterman expected returns.

Feeding raw historical means into a mean-variance optimiser produces the
familiar garbage-in corner solutions. Black & Litterman (1992) fix the
*input*: start from the excess returns implied by market equilibrium
(reverse optimisation of the market portfolio), then tilt them toward the
investor's views in proportion to the confidence in each view.

Equilibrium prior::

    π = δ · Σ · w_mkt                      (δ = risk aversion)

Posterior mean and estimation covariance (He & Litterman, 1999), with
``P`` the k×n view-pick matrix, ``q`` the view returns and ``Ω`` the view
uncertainty (default ``τ · diag(P Σ Pᵀ)``)::

    μ  = π + τΣPᵀ (PτΣPᵀ + Ω)⁻¹ (q − Pπ)
    M  = τΣ − τΣPᵀ (PτΣPᵀ + Ω)⁻¹ PτΣ
    Σ* = Σ + M                              (posterior predictive covariance)

With the default He-Litterman ``Ω ∝ τ``, the posterior mean is independent
of ``τ``; pass an explicit ``omega`` to control view strength directly.
The outputs drop straight into the optimisers of this package (e.g.
``min_variance_weights(returns, cov=result.covariance)``). Pure numpy.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.portfolio.optimizer import _covariance_matrix


@dataclass
class BlackLittermanResult:
    """Posterior of the Black-Litterman model.

    Attributes:
        expected_returns: Posterior per-period expected returns μ.
        covariance: Posterior predictive covariance Σ + M (tickers × tickers).
        implied_returns: The equilibrium prior π = δ Σ w_mkt.
    """

    expected_returns: pd.Series
    covariance: pd.DataFrame
    implied_returns: pd.Series


def black_litterman(
    returns: pd.DataFrame,
    market_weights: pd.Series | Mapping[str, float],
    views: pd.DataFrame | None = None,
    view_returns: pd.Series | np.ndarray | list[float] | None = None,
    cov: pd.DataFrame | np.ndarray | None = None,
    risk_aversion: float = 2.5,
    tau: float = 0.05,
    omega: np.ndarray | None = None,
) -> BlackLittermanResult:
    """Blend equilibrium-implied returns with investor views.

    Args:
        returns: DataFrame of asset returns (columns = tickers), used for
            the covariance estimate and for labelling.
        market_weights: Market-capitalisation weights per ticker; they are
            normalised to sum to 1 internally.
        views: View-pick matrix ``P`` — one row per view, columns exactly
            the tickers of ``returns`` (any order); e.g. a row with 1 on
            one asset is an absolute view, 1/-1 a relative one. ``None``
            (with ``view_returns=None``) returns the pure equilibrium.
        view_returns: Expected per-period return ``q`` of each view row.
        cov: Optional pre-computed covariance matrix.
        risk_aversion: Equilibrium risk-aversion δ (> 0).
        tau: Uncertainty scale of the prior (> 0).
        omega: Optional k×k view-uncertainty matrix; defaults to the
            He-Litterman diagonal ``τ · diag(P Σ Pᵀ)``.

    Returns:
        A :class:`BlackLittermanResult`.

    Raises:
        ValueError: If parameters are non-positive, the weights or views
            do not match the tickers, or the view inputs are inconsistent.
    """
    if risk_aversion <= 0:
        raise ValueError(f"risk_aversion must be > 0, got {risk_aversion}.")
    if tau <= 0:
        raise ValueError(f"tau must be > 0, got {tau}.")
    if (views is None) != (view_returns is None):
        raise ValueError("views and view_returns must be provided together.")

    tickers = list(returns.columns)
    sigma = _covariance_matrix(returns, cov)

    weights = pd.Series(dict(market_weights)).reindex(tickers)
    if weights.isna().any():
        missing = [t for t in tickers if pd.isna(weights[t])]
        raise ValueError(f"market_weights missing tickers {missing}.")
    total = float(weights.sum())
    if total <= 0:
        raise ValueError(f"market_weights must sum to > 0, got {total}.")
    w_mkt = weights.to_numpy(dtype=float) / total

    pi = risk_aversion * sigma @ w_mkt

    if views is None:
        mu = pi
        estimation_cov = tau * sigma
    else:
        if set(views.columns) != set(tickers):
            raise ValueError("views columns must exactly match the return tickers.")
        pick = views[tickers].to_numpy(dtype=float)  # align column order
        q = np.asarray(view_returns, dtype=float).ravel()
        if len(q) != pick.shape[0]:
            raise ValueError(f"view_returns has {len(q)} entries for {pick.shape[0]} view rows.")

        tau_sigma = tau * sigma
        if omega is None:
            omega_matrix = np.diag(np.diag(pick @ tau_sigma @ pick.T))
        else:
            omega_matrix = np.asarray(omega, dtype=float)
            if omega_matrix.shape != (pick.shape[0], pick.shape[0]):
                raise ValueError(
                    f"omega must have shape {(pick.shape[0], pick.shape[0])}, "
                    f"got {omega_matrix.shape}."
                )

        blend = pick @ tau_sigma @ pick.T + omega_matrix
        mu = pi + tau_sigma @ pick.T @ np.linalg.solve(blend, q - pick @ pi)
        estimation_cov = tau_sigma - tau_sigma @ pick.T @ np.linalg.solve(blend, pick @ tau_sigma)

    posterior_cov = sigma + estimation_cov
    return BlackLittermanResult(
        expected_returns=pd.Series(mu, index=tickers, name="bl_returns"),
        covariance=pd.DataFrame(posterior_cov, index=tickers, columns=tickers),
        implied_returns=pd.Series(pi, index=tickers, name="implied_returns"),
    )
