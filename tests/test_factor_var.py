"""Tests for factor-model parametric VaR."""

import numpy as np
import pandas as pd
import pytest

from src.risk import factor_model_var


def _single_factor_inputs() -> tuple[pd.Series, pd.DataFrame, pd.DataFrame, pd.Series]:
    weights = pd.Series({"aaa": 1.0})
    betas = pd.DataFrame({"mkt": [1.0]}, index=["aaa"])
    factor_cov = pd.DataFrame([[0.0001]], index=["mkt"], columns=["mkt"])  # 1% vol
    idio = pd.Series({"aaa": 0.0})
    return weights, betas, factor_cov, idio


def test_unit_beta_portfolio_recovers_factor_vol() -> None:
    result = factor_model_var(*_single_factor_inputs())
    assert result.volatility == pytest.approx(0.01)
    assert result.factor_share == pytest.approx(1.0)
    assert result.idio_variance == 0.0


def test_var_is_gaussian_quantile_of_vol() -> None:
    result = factor_model_var(*_single_factor_inputs(), level=0.05)
    assert result.var == pytest.approx(1.6449 * 0.01, rel=1e-3)
    tighter = factor_model_var(*_single_factor_inputs(), level=0.01)
    assert tighter.var > result.var


def test_idio_only_portfolio() -> None:
    weights = pd.Series({"aaa": 0.5, "bbb": 0.5})
    betas = pd.DataFrame({"mkt": [0.0, 0.0]}, index=["aaa", "bbb"])
    factor_cov = pd.DataFrame([[0.0001]], index=["mkt"], columns=["mkt"])
    idio = pd.Series({"aaa": 0.02, "bbb": 0.02})
    result = factor_model_var(weights, betas, factor_cov, idio)
    # sqrt(0.25*4e-4 + 0.25*4e-4) = 0.01414: idio diversifies
    assert result.volatility == pytest.approx(np.sqrt(2 * 0.25 * 0.0004))
    assert result.factor_share == 0.0


def test_contributions_sum_to_factor_variance() -> None:
    weights = pd.Series({"aaa": 0.6, "bbb": 0.4})
    betas = pd.DataFrame({"mkt": [1.0, 0.8], "value": [0.2, -0.5]}, index=["aaa", "bbb"])
    factor_cov = pd.DataFrame(
        [[0.0001, 0.00002], [0.00002, 0.00005]],
        index=["mkt", "value"],
        columns=["mkt", "value"],
    )
    idio = pd.Series({"aaa": 0.01, "bbb": 0.015})
    result = factor_model_var(weights, betas, factor_cov, idio)
    assert float(result.factor_contributions.sum()) == pytest.approx(result.factor_variance)
    assert list(result.factor_contributions.index) == ["mkt", "value"]
    assert 0.0 < result.factor_share < 1.0


def test_factor_cov_label_order_does_not_matter() -> None:
    weights = pd.Series({"aaa": 1.0})
    betas = pd.DataFrame({"mkt": [1.0], "value": [0.5]}, index=["aaa"])
    cov_a = pd.DataFrame(
        [[0.0001, 0.0], [0.0, 0.0004]], index=["mkt", "value"], columns=["mkt", "value"]
    )
    cov_b = cov_a.loc[["value", "mkt"], ["value", "mkt"]]  # shuffled labels
    idio = pd.Series({"aaa": 0.0})
    a = factor_model_var(weights, betas, cov_a, idio)
    b = factor_model_var(weights, betas, cov_b, idio)
    assert a.volatility == pytest.approx(b.volatility)


def test_hedged_book_has_less_var_than_long_only() -> None:
    betas = pd.DataFrame({"mkt": [1.0, 1.0]}, index=["aaa", "bbb"])
    factor_cov = pd.DataFrame([[0.0001]], index=["mkt"], columns=["mkt"])
    idio = pd.Series({"aaa": 0.005, "bbb": 0.005})
    long_only = factor_model_var(pd.Series({"aaa": 0.5, "bbb": 0.5}), betas, factor_cov, idio)
    hedged = factor_model_var(pd.Series({"aaa": 0.5, "bbb": -0.5}), betas, factor_cov, idio)
    assert hedged.var < long_only.var
    assert hedged.factor_variance == pytest.approx(0.0, abs=1e-18)


def test_bad_inputs_raise() -> None:
    weights, betas, factor_cov, idio = _single_factor_inputs()
    with pytest.raises(ValueError, match="level"):
        factor_model_var(weights, betas, factor_cov, idio, level=1.0)
    with pytest.raises(ValueError, match="betas missing"):
        factor_model_var(pd.Series({"zzz": 1.0}), betas, factor_cov, idio)
    with pytest.raises(ValueError, match="idio_vols missing"):
        factor_model_var(weights, betas, factor_cov, pd.Series({"zzz": 0.01}))
    with pytest.raises(ValueError, match="factor_cov labels"):
        bad_cov = pd.DataFrame([[0.0001]], index=["x"], columns=["x"])
        factor_model_var(weights, betas, bad_cov, idio)
    with pytest.raises(ValueError, match="finite"):
        factor_model_var(pd.Series({"aaa": np.nan}), betas, factor_cov, idio)
    with pytest.raises(ValueError, match=">= 0"):
        factor_model_var(weights, betas, factor_cov, pd.Series({"aaa": -0.01}))
