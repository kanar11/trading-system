"""Tests for the Black-Litterman model."""

import numpy as np
import pandas as pd
import pytest

from src.portfolio import black_litterman, min_variance_weights


def _returns(n: int = 300, seed: int = 12) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2021-01-04", periods=n, freq="B")
    data = {
        "aaa": rng.normal(0.0004, 0.010, n),
        "bbb": rng.normal(0.0002, 0.008, n),
        "ccc": rng.normal(0.0006, 0.015, n),
    }
    return pd.DataFrame(data, index=idx)


_WEIGHTS = {"aaa": 0.5, "bbb": 0.3, "ccc": 0.2}


def test_no_views_returns_pure_equilibrium() -> None:
    df = _returns()
    result = black_litterman(df, _WEIGHTS, risk_aversion=2.5, tau=0.05)
    assert np.allclose(result.expected_returns.to_numpy(), result.implied_returns.to_numpy())
    assert list(result.expected_returns.index) == ["aaa", "bbb", "ccc"]


def test_implied_returns_match_reverse_optimisation() -> None:
    df = _returns()
    result = black_litterman(df, _WEIGHTS, risk_aversion=3.0, cov=df.cov())
    sigma = df.cov().to_numpy() + 1e-8 * np.eye(3)  # optimizer ridge
    w = np.array([0.5, 0.3, 0.2])
    assert np.allclose(result.implied_returns.to_numpy(), 3.0 * sigma @ w)


def test_confident_absolute_view_pins_the_asset() -> None:
    df = _returns()
    views = pd.DataFrame([[1.0, 0.0, 0.0]], columns=["aaa", "bbb", "ccc"])
    target = 0.002
    result = black_litterman(
        df, _WEIGHTS, views=views, view_returns=[target], omega=np.array([[1e-12]])
    )
    assert result.expected_returns["aaa"] == pytest.approx(target, abs=1e-5)


def test_relative_view_moves_the_spread_toward_q() -> None:
    df = _returns()
    views = pd.DataFrame([[1.0, -1.0, 0.0]], columns=["aaa", "bbb", "ccc"])
    q = 0.003
    base = black_litterman(df, _WEIGHTS)
    tilted = black_litterman(df, _WEIGHTS, views=views, view_returns=[q])
    prior_spread = base.implied_returns["aaa"] - base.implied_returns["bbb"]
    post_spread = tilted.expected_returns["aaa"] - tilted.expected_returns["bbb"]
    assert prior_spread < post_spread < q  # pulled toward but not past the view


def test_fixed_omega_tau_to_zero_recovers_prior() -> None:
    df = _returns()
    views = pd.DataFrame([[1.0, 0.0, 0.0]], columns=["aaa", "bbb", "ccc"])
    omega = np.array([[1e-4]])
    tiny_tau = black_litterman(
        df, _WEIGHTS, views=views, view_returns=[0.01], tau=1e-8, omega=omega
    )
    assert np.allclose(
        tiny_tau.expected_returns.to_numpy(),
        tiny_tau.implied_returns.to_numpy(),
        atol=1e-6,
    )


def test_view_column_order_does_not_matter() -> None:
    df = _returns()
    a = black_litterman(
        df,
        _WEIGHTS,
        views=pd.DataFrame([[1.0, -1.0, 0.0]], columns=["aaa", "bbb", "ccc"]),
        view_returns=[0.002],
    )
    b = black_litterman(
        df,
        _WEIGHTS,
        views=pd.DataFrame([[0.0, -1.0, 1.0]], columns=["ccc", "bbb", "aaa"]),
        view_returns=[0.002],
    )
    assert np.allclose(a.expected_returns.to_numpy(), b.expected_returns.to_numpy())


def test_market_weight_scale_is_irrelevant() -> None:
    df = _returns()
    doubled = {k: 2 * v for k, v in _WEIGHTS.items()}
    a = black_litterman(df, _WEIGHTS)
    b = black_litterman(df, doubled)
    assert np.allclose(a.implied_returns.to_numpy(), b.implied_returns.to_numpy())


def test_posterior_covariance_is_symmetric_psd_and_usable() -> None:
    df = _returns()
    views = pd.DataFrame([[1.0, 0.0, -1.0]], columns=["aaa", "bbb", "ccc"])
    result = black_litterman(df, _WEIGHTS, views=views, view_returns=[0.001])
    cov = result.covariance.to_numpy()
    assert np.allclose(cov, cov.T)
    assert np.linalg.eigvalsh(cov).min() > 0
    weights = min_variance_weights(df, cov=result.covariance)
    assert weights.sum() == pytest.approx(1.0)


def test_bad_inputs_raise() -> None:
    df = _returns()
    views = pd.DataFrame([[1.0, 0.0, 0.0]], columns=["aaa", "bbb", "ccc"])
    with pytest.raises(ValueError, match="risk_aversion"):
        black_litterman(df, _WEIGHTS, risk_aversion=0.0)
    with pytest.raises(ValueError, match="tau"):
        black_litterman(df, _WEIGHTS, tau=0.0)
    with pytest.raises(ValueError, match="together"):
        black_litterman(df, _WEIGHTS, views=views)
    with pytest.raises(ValueError, match="missing tickers"):
        black_litterman(df, {"aaa": 0.5, "bbb": 0.5})
    with pytest.raises(ValueError, match="columns"):
        bad = pd.DataFrame([[1.0, 0.0]], columns=["aaa", "bbb"])
        black_litterman(df, _WEIGHTS, views=bad, view_returns=[0.01])
    with pytest.raises(ValueError, match="view_returns"):
        black_litterman(df, _WEIGHTS, views=views, view_returns=[0.01, 0.02])
    with pytest.raises(ValueError, match="omega"):
        black_litterman(df, _WEIGHTS, views=views, view_returns=[0.01], omega=np.eye(2))
