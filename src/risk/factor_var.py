"""Factor-model (parametric) portfolio Value-at-Risk.

Historical VaR needs a long return history of the *current* portfolio —
which a book that was rebalanced yesterday does not have. The industry
answer is the factor model: express each asset as loadings ``B`` on a few
systematic factors plus an idiosyncratic residual, and derive portfolio
risk from the factor covariance::

    σ²_p = wᵀ (B Σ_f Bᵀ + D) w,     D = diag(σ²_idio)

VaR is then the Gaussian quantile of that volatility. The decomposition
falls out for free: how much of the variance is systematic (and which
factor drives it) versus idiosyncratic — the numbers a risk report leads
with. Betas can come straight from
:func:`src.reporting.attribution.factor_regression` per asset. Pure
numpy/pandas; normality caveat as for
:func:`src.risk.metrics.parametric_var`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class FactorVaRResult:
    """Factor-model risk decomposition of a portfolio.

    Attributes:
        var: Parametric VaR at the requested level (positive loss number,
            per period).
        volatility: Portfolio volatility per period.
        factor_variance: Systematic variance ``(Bᵀw)ᵀ Σ_f (Bᵀw)``.
        idio_variance: Idiosyncratic variance ``Σ w²σ²_idio``.
        factor_share: Systematic fraction of total variance in [0, 1].
        factor_contributions: Per-factor variance contribution
            ``(Bᵀw)_j · (Σ_f Bᵀw)_j`` — sums to ``factor_variance``.
    """

    var: float
    volatility: float
    factor_variance: float
    idio_variance: float
    factor_share: float
    factor_contributions: pd.Series


def factor_model_var(
    weights: pd.Series,
    betas: pd.DataFrame,
    factor_cov: pd.DataFrame,
    idio_vols: pd.Series,
    level: float = 0.05,
) -> FactorVaRResult:
    """Compute parametric VaR of a portfolio through a factor model.

    Args:
        weights: Portfolio weights per asset (fraction-of-equity units;
            shorts negative).
        betas: Loadings matrix, rows = assets, columns = factors; must
            cover every asset in ``weights``.
        factor_cov: Factor covariance matrix (factors × factors, same
            labels as the beta columns, any order).
        idio_vols: Per-asset idiosyncratic volatility (per period, >= 0);
            must cover every asset in ``weights``.
        level: Tail probability (0.05 = 95% VaR).

    Returns:
        A :class:`FactorVaRResult`.

    Raises:
        ValueError: If labels are missing/mismatched, values are not
            finite, ``idio_vols`` is negative, or ``level`` is out of
            (0, 1).
    """
    if not 0 < level < 1:
        raise ValueError(f"level must be in (0, 1), got {level}.")

    assets = list(weights.index)
    missing_beta = [a for a in assets if a not in betas.index]
    if missing_beta:
        raise ValueError(f"betas missing assets {missing_beta}.")
    missing_idio = [a for a in assets if a not in idio_vols.index]
    if missing_idio:
        raise ValueError(f"idio_vols missing assets {missing_idio}.")

    factors = list(betas.columns)
    if set(factor_cov.index) != set(factors) or set(factor_cov.columns) != set(factors):
        raise ValueError("factor_cov labels must match the beta columns.")

    w = weights.to_numpy(dtype=float)
    loadings = betas.loc[assets, factors].to_numpy(dtype=float)
    sigma_f = factor_cov.loc[factors, factors].to_numpy(dtype=float)
    idio = idio_vols.loc[assets].to_numpy(dtype=float)

    stacked = np.concatenate([w, loadings.ravel(), sigma_f.ravel(), idio])
    if not np.isfinite(stacked).all():
        raise ValueError("weights, betas, factor_cov and idio_vols must be finite.")
    if (idio < 0).any():
        raise ValueError("idio_vols must be >= 0.")

    exposure = loadings.T @ w  # portfolio loading per factor
    sigma_exposure = sigma_f @ exposure
    contributions = exposure * sigma_exposure
    factor_variance = float(exposure @ sigma_exposure)
    idio_variance = float((w**2 * idio**2).sum())
    total_variance = factor_variance + idio_variance
    volatility = float(np.sqrt(max(total_variance, 0.0)))

    from src.validation.stat_tests import _norm_quantile

    var = float(-_norm_quantile(level)) * volatility
    return FactorVaRResult(
        var=var,
        volatility=volatility,
        factor_variance=factor_variance,
        idio_variance=idio_variance,
        factor_share=factor_variance / total_variance if total_variance > 0 else 0.0,
        factor_contributions=pd.Series(contributions, index=factors, name="variance"),
    )
