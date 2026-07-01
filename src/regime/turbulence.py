"""Financial Turbulence Index (Kritzman & Li, 2010).

A regime measure built on the Mahalanobis distance of a multi-asset return
vector from its "normal" distribution:

    d_t = (y_t − μ)ᵀ Σ⁻¹ (y_t − μ)

Unlike a simple volatility spike, turbulence rises both when returns are
unusually *large* and when they violate the *usual correlation structure*
(e.g. assets that normally move together suddenly diverge). Historically it
spikes around crises and tends to persist, which makes it useful as a
risk-off / regime filter.

Pure numpy/pandas; inputs are never mutated.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def financial_turbulence(
    returns: pd.DataFrame,
    mean: pd.Series | np.ndarray | None = None,
    cov: pd.DataFrame | np.ndarray | None = None,
    ridge: float = 1e-8,
) -> pd.Series:
    """Per-period financial turbulence (Mahalanobis distance).

    The "normal" distribution is described by ``mean`` and ``cov``; when either
    is omitted it is estimated from ``returns`` itself (a full-sample view — for
    a look-ahead-free signal pass a trailing estimate instead).

    Args:
        returns: DataFrame of asset returns (rows = periods, columns = assets).
        mean: Optional mean vector (length = n assets). Estimated if None.
        cov: Optional covariance matrix (n x n). Estimated if None.
        ridge: Small value added to the covariance diagonal for numerical
            stability before inversion.

    Returns:
        Series of turbulence values indexed like ``returns`` (>= 0; NaN for any
        row containing NaN).

    Raises:
        ValueError: If ``returns`` has no columns, or a supplied ``mean`` /
            ``cov`` does not match the number of assets.
    """
    n_assets = returns.shape[1]
    if n_assets == 0:
        raise ValueError("returns must have at least one column")

    mu = returns.mean().to_numpy() if mean is None else np.asarray(mean, dtype=float)
    if mu.shape != (n_assets,):
        raise ValueError(f"mean has length {mu.shape} but returns has {n_assets} columns")

    if cov is None:
        cov_arr = returns.cov().to_numpy()
    elif isinstance(cov, pd.DataFrame):
        cov_arr = cov.to_numpy()
    else:
        cov_arr = np.asarray(cov, dtype=float)
    if cov_arr.shape != (n_assets, n_assets):
        raise ValueError(f"cov has shape {cov_arr.shape} but returns has {n_assets} columns")

    cov_arr = cov_arr + ridge * np.eye(n_assets)
    try:
        inv = np.linalg.inv(cov_arr)
    except np.linalg.LinAlgError:
        inv = np.linalg.pinv(cov_arr)

    diff = returns.to_numpy(dtype=float) - mu
    d = np.einsum("ij,jk,ik->i", diff, inv, diff)
    return pd.Series(d, index=returns.index, name="turbulence")


def turbulent_periods(turbulence: pd.Series, quantile: float = 0.9) -> pd.Series:
    """Boolean mask of periods whose turbulence exceeds a quantile threshold.

    Args:
        turbulence: Turbulence series from :func:`financial_turbulence`.
        quantile: Threshold quantile in [0, 1); periods strictly above the
            corresponding turbulence value are flagged. 0.9 flags roughly the
            most turbulent 10% of periods.

    Returns:
        Boolean Series aligned to ``turbulence`` (True where turbulent).

    Raises:
        ValueError: If ``quantile`` is not in [0, 1).
    """
    if not 0.0 <= quantile < 1.0:
        raise ValueError(f"quantile must be in [0, 1), got {quantile}")
    threshold = turbulence.quantile(quantile)
    mask: pd.Series = turbulence > threshold
    return mask.rename("turbulent")
