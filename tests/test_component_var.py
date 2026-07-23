"""Tests for component / marginal VaR (Euler risk decomposition)."""

import numpy as np
import pandas as pd
import pytest

from src.risk import component_var, marginal_var
from src.risk.metrics import parametric_var


def _cov(values: list[list[float]], labels: list[str]) -> pd.DataFrame:
    return pd.DataFrame(values, index=labels, columns=labels)


def _portfolio() -> tuple[pd.Series, pd.DataFrame]:
    weights = pd.Series({"aaa": 0.6, "bbb": 0.4})
    cov = _cov([[0.0004, 0.0001], [0.0001, 0.0009]], ["aaa", "bbb"])
    return weights, cov


def _portfolio_var(weights: pd.Series, cov: pd.DataFrame, level: float = 0.05) -> float:
    w = weights.to_numpy()
    sigma = cov.loc[list(weights.index), list(weights.index)].to_numpy()
    from src.validation.stat_tests import _norm_quantile

    return float(-_norm_quantile(level)) * float(np.sqrt(w @ sigma @ w))


def test_components_sum_exactly_to_portfolio_var() -> None:
    weights, cov = _portfolio()
    components = component_var(weights, cov)
    assert components.name == "component_var"
    assert float(components.sum()) == pytest.approx(_portfolio_var(weights, cov))


def test_euler_identity_holds_at_any_level() -> None:
    weights, cov = _portfolio()
    for level in (0.01, 0.05, 0.10):
        total = float(component_var(weights, cov, level=level).sum())
        assert total == pytest.approx(_portfolio_var(weights, cov, level=level))


def test_component_is_weight_times_marginal() -> None:
    weights, cov = _portfolio()
    marginal = marginal_var(weights, cov)
    components = component_var(weights, cov)
    assert np.allclose(components.to_numpy(), (weights * marginal).to_numpy())
    assert marginal.name == "marginal_var"


def test_single_asset_matches_the_scalar_var() -> None:
    # one asset at full weight: its component IS the portfolio VaR
    weights = pd.Series({"aaa": 1.0})
    cov = _cov([[0.0004]], ["aaa"])  # 2% vol
    components = component_var(weights, cov, level=0.05)
    assert components["aaa"] == pytest.approx(1.6449 * 0.02, rel=1e-3)


def test_riskier_asset_carries_the_bigger_budget() -> None:
    # equal weights, but bbb has 3x the variance -> it must dominate the budget
    weights = pd.Series({"aaa": 0.5, "bbb": 0.5})
    cov = _cov([[0.0004, 0.0], [0.0, 0.0012]], ["aaa", "bbb"])
    components = component_var(weights, cov)
    assert components["bbb"] > components["aaa"]


def test_hedge_shows_a_negative_component() -> None:
    # a short in a perfectly correlated asset hedges the book
    weights = pd.Series({"aaa": 1.0, "hedge": -0.5})
    cov = _cov([[0.0004, 0.0004], [0.0004, 0.0004]], ["aaa", "hedge"])
    components = component_var(weights, cov)
    assert components["hedge"] < 0
    assert float(components.sum()) == pytest.approx(_portfolio_var(weights, cov))


def test_zero_volatility_portfolio_is_all_zero() -> None:
    # a perfectly offsetting book has no risk to allocate
    weights = pd.Series({"aaa": 1.0, "bbb": -1.0})
    cov = _cov([[0.0004, 0.0004], [0.0004, 0.0004]], ["aaa", "bbb"])
    assert np.allclose(marginal_var(weights, cov).to_numpy(), 0.0)
    assert np.allclose(component_var(weights, cov).to_numpy(), 0.0)


def test_matches_parametric_var_on_a_single_asset_series() -> None:
    # cross-check against the existing scalar VaR on a synthetic series
    rng = np.random.default_rng(7)
    returns = pd.Series(rng.normal(0.0, 0.02, 20_000))
    weights = pd.Series({"aaa": 1.0})
    cov = _cov([[float(returns.var(ddof=1))]], ["aaa"])
    decomposed = float(component_var(weights, cov, level=0.05).sum())
    # parametric_var includes the (near-zero) mean drift; the decomposition
    # is the zero-mean Gaussian quantile, so they agree to a small tolerance
    assert decomposed == pytest.approx(parametric_var(returns, level=0.05), rel=0.05)


def test_bad_inputs_raise() -> None:
    weights, cov = _portfolio()
    with pytest.raises(ValueError, match="level"):
        component_var(weights, cov, level=1.0)
    with pytest.raises(ValueError, match="cov missing"):
        component_var(pd.Series({"zzz": 1.0}), cov)
    with pytest.raises(ValueError, match="finite"):
        marginal_var(pd.Series({"aaa": np.nan, "bbb": 1.0}), cov)
