"""Tests for Cornish-Fisher (modified) VaR."""

import numpy as np
import pandas as pd
import pytest

from src.risk.metrics import cornish_fisher_var, parametric_var


def _normal_returns(n: int = 20_000, seed: int = 0) -> pd.Series:
    rng = np.random.default_rng(seed)
    return pd.Series(rng.normal(0.0, 0.01, n))


def test_gaussian_data_matches_parametric_var() -> None:
    returns = _normal_returns()
    modified = cornish_fisher_var(returns, level=0.05)
    gaussian = parametric_var(returns, level=0.05)
    assert modified == pytest.approx(gaussian, rel=0.02)


def test_negative_skew_raises_the_loss_estimate() -> None:
    rng = np.random.default_rng(1)
    base = rng.normal(0.0, 0.01, 20_000)
    crashes = -np.abs(rng.normal(0.0, 0.04, 500))  # a fat left tail
    returns = pd.Series(np.concatenate([base, crashes]))
    assert cornish_fisher_var(returns, level=0.01) > parametric_var(returns, level=0.01)


def test_skew_direction_isolated_by_mirroring() -> None:
    # mirroring a demeaned sample flips the skew while keeping mean (0),
    # sigma and kurtosis identical — the pure skew effect on the left tail
    rng = np.random.default_rng(2)
    base = rng.normal(0.0, 0.01, 20_000)
    crashes = -np.abs(rng.normal(0.0, 0.04, 500))
    sample = pd.Series(np.concatenate([base, crashes]))
    demeaned = sample - sample.mean()
    negatively_skewed = cornish_fisher_var(demeaned, level=0.01)
    positively_skewed = cornish_fisher_var(-demeaned, level=0.01)
    assert negatively_skewed > positively_skewed


def test_tighter_level_means_bigger_var() -> None:
    returns = _normal_returns(seed=3)
    assert cornish_fisher_var(returns, level=0.01) > cornish_fisher_var(returns, level=0.05)


def test_matches_the_favre_galeano_formula() -> None:
    from src.risk.metrics import kurtosis, skewness
    from src.validation.stat_tests import _norm_quantile

    rng = np.random.default_rng(4)
    returns = pd.Series(rng.standard_t(df=4, size=5_000) * 0.005)
    z = _norm_quantile(0.05)
    s, k = skewness(returns), kurtosis(returns)
    z_cf = z + (z**2 - 1) * s / 6 + (z**3 - 3 * z) * k / 24 - (2 * z**3 - 5 * z) * s**2 / 36
    expected = -(float(returns.mean()) + z_cf * float(returns.std(ddof=1)))
    assert cornish_fisher_var(returns, level=0.05) == pytest.approx(expected)


def test_short_series_returns_zero() -> None:
    assert cornish_fisher_var(pd.Series([0.01, -0.02, 0.005])) == 0.0
    assert cornish_fisher_var(pd.Series(dtype=float)) == 0.0


def test_bad_level_raises() -> None:
    with pytest.raises(ValueError, match="level"):
        cornish_fisher_var(_normal_returns(100), level=0.0)
    with pytest.raises(ValueError, match="level"):
        cornish_fisher_var(_normal_returns(100), level=1.0)
