"""Ledoit-Wolf covariance shrinkage.

The sample covariance matrix is the worst input you can feed a portfolio
optimiser when assets are many and observations few: it is noisy, often
ill-conditioned or outright singular (p > n), and mean-variance weights
amplify exactly that noise. Ledoit & Wolf ("A well-conditioned estimator
for large-dimensional covariance matrices", JMVA 2004) fix this by
shrinking the sample matrix toward a scaled-identity target::

    Σ* = δ·m·I + (1 − δ)·S

where ``m`` is the average sample variance and the shrinkage intensity
``δ ∈ [0, 1]`` is estimated *analytically* from the data — no
cross-validation, no tuning. The result is always well-conditioned
(positive definite for any n, p) and trace-preserving.

The returned matrix is a plain labelled DataFrame, so it drops straight
into the ``cov=`` argument of every optimiser in
:mod:`src.portfolio.optimizer` and into :func:`src.portfolio.frontier.
efficient_frontier`. Pure numpy.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class ShrinkageResult:
    """Output of the Ledoit-Wolf estimator.

    Attributes:
        covariance: Shrunk covariance matrix (tickers × tickers).
        shrinkage: Estimated shrinkage intensity δ in [0, 1]; 0 keeps the
            sample matrix, 1 replaces it entirely with the identity target.
        target_variance: The target's common variance ``m`` (average of the
            sample variances, i.e. ``tr(S)/p``).
    """

    covariance: pd.DataFrame
    shrinkage: float
    target_variance: float


def ledoit_wolf_covariance(returns: pd.DataFrame) -> ShrinkageResult:
    """Estimate a well-conditioned covariance matrix via Ledoit-Wolf.

    Implements the scaled-identity shrinkage of Ledoit & Wolf (2004):
    with ``S`` the (1/n-normalised) sample covariance of the demeaned
    returns and ``<A, B> = tr(ABᵀ)/p`` the scaled Frobenius inner product,

    * ``m  = <S, I>``           — target variance,
    * ``d² = ‖S − mI‖²``        — total dispersion of S around the target,
    * ``b² = min(β̄², d²)``     — estimation-error part, where ``β̄²`` is
      the average squared distance of the per-observation outer products
      from S, divided by n,
    * ``δ  = b²/d²``            — shrinkage intensity.

    Args:
        returns: DataFrame of asset returns (rows = observations, columns =
            tickers), at least 2 rows, NaN-free.

    Returns:
        A :class:`ShrinkageResult`; ``covariance`` is positive definite even
        when there are fewer observations than assets.

    Raises:
        ValueError: If ``returns`` has no columns, fewer than 2 rows, or
            contains NaNs.
    """
    if returns.shape[1] == 0:
        raise ValueError("returns must have at least one column.")
    if returns.shape[0] < 2:
        raise ValueError(f"returns must have at least 2 rows, got {returns.shape[0]}.")

    x = returns.to_numpy(dtype=float)
    if np.isnan(x).any():
        raise ValueError("returns must not contain NaNs.")

    n, p = x.shape
    x = x - x.mean(axis=0)
    sample = x.T @ x / n  # Ledoit-Wolf normalise by n, not n-1

    m = float(np.trace(sample)) / p
    delta_sq = float(((sample - m * np.eye(p)) ** 2).sum()) / p

    if delta_sq <= 0.0:
        # S already equals the target (e.g. perfectly homoskedastic,
        # uncorrelated data) — nothing to shrink
        cov = pd.DataFrame(sample, index=returns.columns, columns=returns.columns)
        return ShrinkageResult(covariance=cov, shrinkage=0.0, target_variance=m)

    # average squared distance of per-observation outer products from S
    beta_bar_sq = 0.0
    for row in x:
        diff = np.outer(row, row) - sample
        beta_bar_sq += float((diff**2).sum()) / p
    beta_bar_sq /= n * n

    b_sq = min(beta_bar_sq, delta_sq)
    shrinkage = b_sq / delta_sq

    shrunk = shrinkage * m * np.eye(p) + (1.0 - shrinkage) * sample
    cov = pd.DataFrame(shrunk, index=returns.columns, columns=returns.columns)
    return ShrinkageResult(covariance=cov, shrinkage=shrinkage, target_variance=m)
