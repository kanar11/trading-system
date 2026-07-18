"""Tests for EWMA (RiskMetrics) covariance estimation."""

import numpy as np
import pandas as pd
import pytest

from src.portfolio import ewma_covariance, min_variance_weights


def _returns(n: int = 400, seed: int = 17) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2022-01-03", periods=n, freq="B")
    return pd.DataFrame({"aaa": rng.normal(0, 0.01, n), "bbb": rng.normal(0, 0.02, n)}, index=idx)


def test_shape_labels_and_symmetry() -> None:
    df = _returns()
    cov = ewma_covariance(df)
    assert cov.shape == (2, 2)
    assert list(cov.index) == ["aaa", "bbb"]
    assert np.allclose(cov.to_numpy(), cov.to_numpy().T)


def test_high_decay_approaches_sample_covariance() -> None:
    df = _returns(n=200)
    ewma = ewma_covariance(df, decay=0.9999, demean=True)
    sample = df.cov(ddof=0)
    assert np.allclose(ewma.to_numpy(), sample.to_numpy(), rtol=0.02)


def test_tracks_a_volatility_regime_shift() -> None:
    rng = np.random.default_rng(3)
    calm = rng.normal(0, 0.005, size=(250, 1))
    wild = rng.normal(0, 0.03, size=(60, 1))
    df = pd.DataFrame(np.vstack([calm, wild]), columns=["a"])
    ewma_var = float(ewma_covariance(df, decay=0.94).iloc[0, 0])
    sample_var = float(df.var(ddof=0).iloc[0])
    wild_var = float(np.var(wild))
    # EWMA sits near the recent wild regime; the sample estimate lags far below
    assert abs(ewma_var - wild_var) < abs(sample_var - wild_var)
    assert ewma_var > 2 * sample_var


def test_positive_semi_definite() -> None:
    rng = np.random.default_rng(8)
    df = pd.DataFrame(rng.normal(0, 0.01, size=(50, 6)))
    cov = ewma_covariance(df, decay=0.94)
    eigenvalues = np.linalg.eigvalsh(cov.to_numpy())
    assert eigenvalues.min() > -1e-12


def test_feeds_the_optimizers() -> None:
    df = _returns()
    cov = ewma_covariance(df, decay=0.97)
    weights = min_variance_weights(df, cov=cov)
    assert weights.sum() == pytest.approx(1.0)
    # min-variance prefers the lower-vol asset under any sane estimate
    assert weights["aaa"] > weights["bbb"]


def test_newest_observation_dominates_at_low_decay() -> None:
    df = pd.DataFrame({"a": [0.0, 0.0, 0.10]})
    cov = ewma_covariance(df, decay=0.01)
    # weights ~ [1e-4, 1e-2, ~0.99] -> variance ~ 0.99 * 0.1^2
    assert float(cov.iloc[0, 0]) == pytest.approx(0.01, rel=0.05)


def test_bad_inputs_raise() -> None:
    df = _returns(10)
    with pytest.raises(ValueError, match="decay"):
        ewma_covariance(df, decay=1.0)
    with pytest.raises(ValueError, match="decay"):
        ewma_covariance(df, decay=0.0)
    with pytest.raises(ValueError, match="column"):
        ewma_covariance(df[[]])
    with pytest.raises(ValueError, match="rows"):
        ewma_covariance(df.iloc[:1])
    bad = df.copy()
    bad.iloc[2, 0] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        ewma_covariance(bad)
