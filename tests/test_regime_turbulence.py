"""Tests for the Financial Turbulence Index."""

import numpy as np
import pandas as pd
import pytest

from src.regime import financial_turbulence, turbulent_periods


def _returns(n: int = 200, n_assets: int = 3, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    data = rng.normal(0.0, 0.01, size=(n, n_assets))
    return pd.DataFrame(data, index=idx, columns=[f"a{i}" for i in range(n_assets)])


def test_index_and_name_align() -> None:
    df = _returns()
    out = financial_turbulence(df)
    assert out.name == "turbulence"
    assert out.index.equals(df.index)
    assert len(out) == len(df)


def test_non_negative() -> None:
    out = financial_turbulence(_returns())
    assert (out.dropna() >= 0).all()


def test_single_asset_equals_squared_zscore() -> None:
    idx = pd.date_range("2020-01-01", periods=6, freq="B")
    x = pd.DataFrame({"a": [0.01, -0.02, 0.03, -0.01, 0.02, -0.03]}, index=idx)
    out = financial_turbulence(x, ridge=0.0)
    mu = float(x["a"].mean())
    var = float(x["a"].var(ddof=1))  # pandas cov uses ddof=1
    expected = ((x["a"] - mu) ** 2 / var).to_numpy()
    assert np.allclose(out.to_numpy(), expected)


def test_correlation_breakdown_dominates() -> None:
    # normal distribution: two strongly *positively* correlated assets
    cov = np.array([[1.0, 0.9], [0.9, 1.0]])
    mean = np.array([0.0, 0.0])
    # row 0 respects the correlation (both up); row 1 breaks it (opposite signs)
    obs = pd.DataFrame({"a": [3.0, 3.0], "b": [3.0, -3.0]})
    out = financial_turbulence(obs, mean=mean, cov=cov, ridge=0.0)
    # a same-magnitude move that violates the usual correlation is far more
    # turbulent than one that respects it
    assert out.iloc[1] > out.iloc[0] * 10


def test_outlier_row_is_most_turbulent() -> None:
    df = _returns()
    df.iloc[100] = [0.5, -0.4, 0.45]  # a violent joint dislocation
    out = financial_turbulence(df)
    assert out.idxmax() == df.index[100]


def test_accepts_explicit_mean_and_cov_series() -> None:
    df = _returns()
    mean = df.mean()
    cov = df.cov()
    out = financial_turbulence(df, mean=mean, cov=cov)
    assert len(out) == len(df)
    assert (out.dropna() >= 0).all()


def test_nan_row_yields_nan() -> None:
    df = _returns(n=50)
    df.iloc[10, 0] = np.nan
    out = financial_turbulence(df)
    assert np.isnan(out.iloc[10])
    assert out.drop(out.index[10]).notna().all()


def test_singular_covariance_uses_pseudoinverse() -> None:
    # perfectly collinear assets -> singular cov; with no ridge np.linalg.inv
    # raises and the pseudo-inverse fallback kicks in
    singular = np.array([[1.0, 1.0], [1.0, 1.0]])
    obs = pd.DataFrame({"a": [1.0, -2.0], "b": [1.0, 3.0]})
    out = financial_turbulence(obs, mean=np.zeros(2), cov=singular, ridge=0.0)
    assert np.isfinite(out).all()
    assert (out >= 0).all()


def test_no_columns_raises() -> None:
    with pytest.raises(ValueError, match="at least one column"):
        financial_turbulence(pd.DataFrame(index=[0, 1, 2]))


def test_mean_shape_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="mean"):
        financial_turbulence(_returns(n_assets=3), mean=np.zeros(2))


def test_cov_shape_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="cov"):
        financial_turbulence(_returns(n_assets=3), cov=np.eye(2))


# --- turbulent_periods -----------------------------------------------------


def test_turbulent_periods_flags_top_decile() -> None:
    out = financial_turbulence(_returns(n=100))
    mask = turbulent_periods(out, quantile=0.9)
    assert mask.dtype == bool
    assert mask.name == "turbulent"
    # ~10% flagged (strict >, so at most 10 of 100)
    assert 5 <= int(mask.sum()) <= 12


def test_turbulent_periods_higher_quantile_flags_fewer() -> None:
    out = financial_turbulence(_returns(n=200))
    assert int(turbulent_periods(out, 0.95).sum()) <= int(turbulent_periods(out, 0.80).sum())


def test_turbulent_periods_bad_quantile_raises() -> None:
    out = financial_turbulence(_returns(n=50))
    with pytest.raises(ValueError, match="quantile"):
        turbulent_periods(out, quantile=1.0)
