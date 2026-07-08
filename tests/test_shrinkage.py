"""Tests for Ledoit-Wolf covariance shrinkage."""

import numpy as np
import pandas as pd
import pytest

from src.portfolio import ledoit_wolf_covariance, min_variance_weights


def _returns(n: int = 250, p: int = 4, seed: int = 11) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    data = rng.normal(0.0, 0.01, size=(n, p))
    return pd.DataFrame(data, index=idx, columns=[f"a{i}" for i in range(p)])


def test_shape_labels_and_symmetry() -> None:
    df = _returns()
    result = ledoit_wolf_covariance(df)
    cov = result.covariance
    assert cov.shape == (4, 4)
    assert list(cov.index) == list(df.columns)
    assert list(cov.columns) == list(df.columns)
    assert np.allclose(cov.to_numpy(), cov.to_numpy().T)


def test_shrinkage_intensity_in_unit_interval() -> None:
    result = ledoit_wolf_covariance(_returns())
    assert 0.0 <= result.shrinkage <= 1.0


def test_reconstruction_from_components() -> None:
    # Sigma* = delta*m*I + (1-delta)*S with S the 1/n sample covariance
    df = _returns(n=60, p=3)
    result = ledoit_wolf_covariance(df)
    x = df.to_numpy() - df.to_numpy().mean(axis=0)
    sample = x.T @ x / len(df)
    expected = (
        result.shrinkage * result.target_variance * np.eye(3) + (1.0 - result.shrinkage) * sample
    )
    assert np.allclose(result.covariance.to_numpy(), expected)


def test_trace_is_preserved() -> None:
    df = _returns(n=100, p=5)
    result = ledoit_wolf_covariance(df)
    x = df.to_numpy() - df.to_numpy().mean(axis=0)
    sample = x.T @ x / len(df)
    assert float(np.trace(result.covariance.to_numpy())) == pytest.approx(float(np.trace(sample)))


def test_positive_definite_even_when_p_exceeds_n() -> None:
    # 6 observations of 20 assets: the sample matrix is singular, the
    # shrunk one must not be
    df = _returns(n=6, p=20, seed=2)
    result = ledoit_wolf_covariance(df)
    eigenvalues = np.linalg.eigvalsh(result.covariance.to_numpy())
    assert eigenvalues.min() > 0
    assert result.shrinkage > 0.0


def test_more_observations_shrink_less() -> None:
    # needs a genuinely heterogeneous true covariance: for iid equal-variance
    # data the identity target IS the truth and delta need not fall with n
    rng = np.random.default_rng(7)
    mix = rng.normal(size=(5, 5)) + 2 * np.eye(5)
    data = rng.normal(0.0, 0.01, size=(2_000, 5)) @ mix
    small = ledoit_wolf_covariance(pd.DataFrame(data[:30]))
    large = ledoit_wolf_covariance(pd.DataFrame(data))
    assert large.shrinkage < small.shrinkage


def test_degenerate_dispersion_returns_sample_unshrunk() -> None:
    # two uncorrelated assets with exactly equal variances: S already equals
    # the m*I target, so d^2 = 0 and nothing is shrunk. 0.25 is exact in
    # binary, so the zero-dispersion case is hit without float dust.
    df = pd.DataFrame({"a": [0.25, -0.25, 0.25, -0.25], "b": [0.25, 0.25, -0.25, -0.25]})
    result = ledoit_wolf_covariance(df)
    assert result.shrinkage == 0.0
    assert result.target_variance == pytest.approx(0.0625)


def test_feeds_the_optimizer() -> None:
    df = _returns()
    result = ledoit_wolf_covariance(df)
    weights = min_variance_weights(df, cov=result.covariance)
    assert weights.sum() == pytest.approx(1.0)
    assert (weights.to_numpy() >= 0).all()


def test_bad_inputs_raise() -> None:
    with pytest.raises(ValueError, match="column"):
        ledoit_wolf_covariance(_returns()[[]])
    with pytest.raises(ValueError, match="rows"):
        ledoit_wolf_covariance(_returns().iloc[:1])
    bad = _returns(n=20)
    bad.iloc[3, 1] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        ledoit_wolf_covariance(bad)
