"""Statistical significance tests for backtest returns.

This module answers two important questions that the basic Sharpe
ratio cannot:

    1. "Is my strategy's Sharpe actually different from zero, given
       sample size and non-normality?"  → :func:`sharpe_ttest`,
       :func:`probabilistic_sharpe_ratio`.

    2. "Given that I ran N parameter combinations, is the *best*
       Sharpe still likely a real effect or just selection bias?"
       → :func:`deflated_sharpe_ratio`.

References:
    Bailey, D.H. & López de Prado, M. (2012, 2014). The Sharpe Ratio
    Efficient Frontier; The Deflated Sharpe Ratio. Journal of
    Portfolio Management.

All routines are pure numpy / pandas — no scipy. The normal CDF is
approximated with ``math.erf`` from the standard library.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

TRADING_DAYS = 252


def _norm_cdf(x: float) -> float:
    """Standard-normal CDF via the error function."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _moments(returns: np.ndarray) -> tuple[float, float, float, float]:
    """Return (mean, std, skew, excess_kurtosis) with ddof=1."""
    n = len(returns)
    mean = float(returns.mean())
    var = float(returns.var(ddof=1))
    std = math.sqrt(var) if var > 0 else 0.0

    if std == 0 or n < 4:
        return mean, std, 0.0, 0.0

    m3 = float(((returns - mean) ** 3).sum() / n)
    m4 = float(((returns - mean) ** 4).sum() / n)
    skew = m3 / std ** 3
    kurt = m4 / std ** 4 - 3.0  # excess kurtosis
    return mean, std, skew, kurt


def _annualised_sharpe(returns: np.ndarray, rf_daily: float = 0.0) -> float:
    mean, std, *_ = _moments(returns)
    if std == 0:
        return 0.0
    return (mean - rf_daily) / std * math.sqrt(TRADING_DAYS)


@dataclass
class SharpeTestResult:
    sharpe_annualised: float
    t_stat: float
    p_value_two_sided: float
    n_obs: int


def sharpe_ttest(returns: pd.Series, rf_daily: float = 0.0) -> SharpeTestResult:
    """t-test for H0: annualised Sharpe == 0.

    Uses the classical Sharpe-ratio standard error
        SE(SR) = sqrt(1 / N) (assuming iid normal returns)
    and reports the two-sided p-value under the asymptotic normal
    approximation. For finite samples this slightly overstates
    significance — see :func:`probabilistic_sharpe_ratio` for a
    higher-moment correction.

    Args:
        returns: Daily return series.
        rf_daily: Daily risk-free rate.

    Returns:
        :class:`SharpeTestResult`.
    """
    r = pd.Series(returns).dropna().values
    n = len(r)
    if n < 2:
        return SharpeTestResult(0.0, 0.0, 1.0, n)

    sr = _annualised_sharpe(r, rf_daily=rf_daily)
    # Sharpe t-stat under iid normal: t = SR * sqrt(N) (where SR is daily)
    sr_daily = sr / math.sqrt(TRADING_DAYS)
    t = sr_daily * math.sqrt(n)
    p_two = 2.0 * (1.0 - _norm_cdf(abs(t)))
    return SharpeTestResult(sr, t, p_two, n)


def probabilistic_sharpe_ratio(
    returns: pd.Series,
    target_sharpe: float = 0.0,
    rf_daily: float = 0.0,
) -> float:
    """Probability that the *true* Sharpe exceeds ``target_sharpe``.

    Bailey & López de Prado (2012) Sharpe-ratio confidence statistic
    that corrects for skewness and excess kurtosis:

        PSR(SR*) = Φ((SR_hat - SR*) * sqrt(N - 1) /
                     sqrt(1 - γ_3 * SR_hat + (γ_4 / 4) * SR_hat²))

    where γ_3 = skew, γ_4 = excess kurtosis, and all Sharpe values
    are *daily*. Returns the resulting probability in [0, 1].

    Args:
        returns: Daily return series.
        target_sharpe: Threshold Sharpe (annualised) to beat.
        rf_daily: Daily risk-free rate.

    Returns:
        Probability ∈ [0, 1].
    """
    r = pd.Series(returns).dropna().values
    n = len(r)
    if n < 4:
        return 0.5

    sr_ann = _annualised_sharpe(r, rf_daily=rf_daily)
    sr_daily = sr_ann / math.sqrt(TRADING_DAYS)
    target_daily = target_sharpe / math.sqrt(TRADING_DAYS)

    _, _, skew, kurt = _moments(r)
    denom = 1.0 - skew * sr_daily + (kurt / 4.0) * sr_daily ** 2
    if denom <= 0:
        return 0.5  # degenerate distribution

    z = (sr_daily - target_daily) * math.sqrt(n - 1) / math.sqrt(denom)
    return _norm_cdf(z)


def deflated_sharpe_ratio(
    returns: pd.Series,
    n_trials: int,
    trial_sharpe_std: float | None = None,
    rf_daily: float = 0.0,
) -> float:
    """Deflated Sharpe Ratio (Bailey & López de Prado, 2014).

    Adjusts the probabilistic-Sharpe statistic for selection bias when
    the reported Sharpe came from the best of ``n_trials`` candidate
    strategies. The deflation is driven by the expected maximum of N
    iid normal trial Sharpes, approximated by the standard
    extreme-value asymptotic:

        E[max] ≈ (1 - γ) * Φ⁻¹(1 - 1/N) + γ * Φ⁻¹(1 - 1/(N e))
        where γ = Euler-Mascheroni constant.

    The target Sharpe in the PSR formula is then set to this expected
    maximum scaled by the cross-trial Sharpe dispersion.

    Args:
        returns: Daily return series of the *winning* strategy.
        n_trials: Number of distinct strategies / parameter combos
            tested.
        trial_sharpe_std: Cross-trial standard deviation of annualised
            Sharpe ratios. If None, defaults to the in-sample Sharpe
            standard error (1 / sqrt(N) * sqrt(252)).
        rf_daily: Daily risk-free rate.

    Returns:
        Deflated PSR ∈ [0, 1]. Values close to 1 → the strategy
        likely has real edge even after accounting for trial inflation.
    """
    r = pd.Series(returns).dropna().values
    n = len(r)
    if n < 4 or n_trials < 1:
        return 0.5

    # expected max of n_trials iid std-normals (Euler-Mascheroni asymptotic)
    if n_trials == 1:
        expected_max = 0.0
    else:
        gamma = 0.5772156649
        emax = (1 - gamma) * _norm_quantile(1 - 1 / n_trials) + gamma * _norm_quantile(
            1 - 1 / (n_trials * math.e)
        )
        expected_max = emax

    if trial_sharpe_std is None:
        # default fallback — standard error of an annualised Sharpe under iid normality
        trial_sharpe_std = math.sqrt(TRADING_DAYS / n)

    target_ann_sharpe = expected_max * trial_sharpe_std
    return probabilistic_sharpe_ratio(
        pd.Series(r), target_sharpe=target_ann_sharpe, rf_daily=rf_daily
    )


def _norm_quantile(p: float) -> float:
    """Approximate inverse normal CDF (rational approximation, Acklam 2003).

    Accurate to ~1e-9 across the unit interval. Avoids a scipy dep.
    """
    if not 0 < p < 1:
        raise ValueError("p must be in (0, 1)")

    # Coefficients
    a = [-3.969683028665376e+01, 2.209460984245205e+02,
         -2.759285104469687e+02, 1.383577518672690e+02,
         -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02,
         -1.556989798598866e+02, 6.680131188771972e+01,
         -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01,
         -2.400758277161838e+00, -2.549732539343734e+00,
         4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01,
         2.445134137142996e+00, 3.754408661907416e+00]

    p_low = 0.02425
    p_high = 1 - p_low

    if p < p_low:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
               ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    if p <= p_high:
        q = p - 0.5
        r = q * q
        return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
               (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)
    q = math.sqrt(-2 * math.log(1 - p))
    return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
           ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
