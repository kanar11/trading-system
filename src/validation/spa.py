"""Hansen's test for Superior Predictive Ability (SPA).

White's Reality Check (:mod:`src.validation.reality_check`) has two known
weaknesses (Hansen, JBES 2005): it compares *raw* means, so one noisy
irrelevant strategy can dominate the max statistic, and it recentres every
strategy to zero mean, so stuffing the candidate set with deeply losing
strategies drags the null distribution out and inflates the p-value. The
SPA test fixes both:

* **studentisation** — each strategy's statistic is ``√T·μ̂_k / ω̂_k``,
  with ``ω̂_k`` estimated from the bootstrap distribution of its mean;
* **threshold recentring** — the "consistent" null keeps the (negative)
  sample mean of strategies that are significantly bad
  (``z_k ≤ −√(2 ln ln T)``) instead of pretending they are competitive.

Three p-values come out, ordered ``lower ≤ consistent ≤ upper``:
``upper`` recentres everything (RC-like, conservative), ``lower`` assumes
every non-positive mean is real (liberal), and ``consistent`` — the
headline number — sits in between with the threshold rule. Pure numpy;
shares the moving-block bootstrap with the Reality Check.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.validation.reality_check import _block_bootstrap_indices

_ZERO_VARIANCE = 1e-15


@dataclass
class SPAResult:
    """Outcome of Hansen's SPA test.

    Attributes:
        p_value: The consistent SPA p-value (the headline number). Small =
            the best strategy beats the benchmark even after accounting
            for the search.
        p_value_lower: Liberal variant (all non-positive means kept).
        p_value_upper: Conservative variant (everything recentred; the
            studentised analogue of White's RC).
        best_strategy: Column index with the highest studentised statistic.
        test_statistic: ``max(max_k √T μ̂_k/ω̂_k, 0)``.
    """

    p_value: float
    p_value_lower: float
    p_value_upper: float
    best_strategy: int
    test_statistic: float


def hansen_spa(
    performance: np.ndarray,
    n_bootstrap: int = 1000,
    block_size: int = 1,
    seed: int | None = None,
) -> SPAResult:
    """Run Hansen's SPA test over a panel of benchmark-relative performance.

    Args:
        performance: (T, N) array of per-period performance for N candidate
            strategies (e.g. excess returns vs the benchmark), exactly as
            for :func:`~src.validation.reality_check.whites_reality_check`.
        n_bootstrap: Number of bootstrap resamples.
        block_size: Moving-block length (1 = iid; > 1 preserves short-range
            autocorrelation).
        seed: RNG seed for reproducibility.

    Returns:
        An :class:`SPAResult`.

    Raises:
        ValueError: If the panel is not 2-D, has fewer than 3 periods, all
            strategies have zero variance, or the bootstrap parameters are
            out of range.
    """
    panel = np.asarray(performance, dtype=float)
    if panel.ndim != 2:
        raise ValueError("performance must be 2-D (T periods x N strategies).")
    n_obs, n_strategies = panel.shape
    if n_strategies < 1:
        raise ValueError("performance must have at least one strategy.")
    if n_obs < 3:
        raise ValueError(f"need at least 3 periods, got {n_obs}.")
    if n_bootstrap < 1:
        raise ValueError(f"n_bootstrap must be >= 1, got {n_bootstrap}.")
    if not 1 <= block_size <= n_obs:
        raise ValueError(f"block_size must be in [1, {n_obs}], got {block_size}.")

    rng = np.random.default_rng(seed)
    means = panel.mean(axis=0)
    root_t = float(np.sqrt(n_obs))

    boot_means = np.empty((n_bootstrap, n_strategies))
    for b in range(n_bootstrap):
        idx = _block_bootstrap_indices(rng, n_obs, block_size)
        boot_means[b] = panel[idx].mean(axis=0)

    # bootstrap estimate of omega_k = std of sqrt(T) * mean_k
    omega = root_t * boot_means.std(axis=0, ddof=1)
    valid = omega > _ZERO_VARIANCE
    if not valid.any():
        raise ValueError("all strategies have zero variance; nothing to test.")

    z_obs = np.full(n_strategies, -np.inf)
    z_obs[valid] = root_t * means[valid] / omega[valid]
    test_statistic = max(float(z_obs[valid].max()), 0.0)
    best = int(np.argmax(z_obs))

    # Hansen's threshold: strategies significantly worse than the benchmark
    # keep their negative mean under the consistent null
    threshold = float(np.sqrt(2.0 * max(np.log(np.log(n_obs)), 0.0)))
    mu_lower = np.minimum(means, 0.0)
    mu_consistent = np.where(z_obs <= -threshold, means, 0.0)
    mu_upper = np.zeros(n_strategies)

    def _p_value(recentred_mean: np.ndarray) -> float:
        shifted = boot_means[:, valid] - means[valid] + recentred_mean[valid]
        z_boot = root_t * shifted / omega[valid]
        stats = np.maximum(z_boot.max(axis=1), 0.0)
        return float((stats >= test_statistic).mean())

    return SPAResult(
        p_value=_p_value(mu_consistent),
        p_value_lower=_p_value(mu_lower),
        p_value_upper=_p_value(mu_upper),
        best_strategy=best,
        test_statistic=test_statistic,
    )
