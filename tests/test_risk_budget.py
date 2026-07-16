"""Tests for risk-budget portfolio weights."""

import numpy as np
import pandas as pd
import pytest

from src.portfolio import risk_budget_weights, risk_contributions, risk_parity_weights


def _returns(n: int = 400, seed: int = 21) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2022-01-03", periods=n, freq="B")
    data = {
        "aaa": rng.normal(0.0, 0.008, n),
        "bbb": rng.normal(0.0, 0.015, n),
        "ccc": rng.normal(0.0, 0.020, n),
    }
    return pd.DataFrame(data, index=idx)


def test_equal_budgets_reproduce_risk_parity() -> None:
    df = _returns()
    budgeted = risk_budget_weights(df, {"aaa": 1.0, "bbb": 1.0, "ccc": 1.0})
    parity = risk_parity_weights(df)
    assert np.allclose(budgeted.to_numpy(), parity.to_numpy())
    assert budgeted.name == "risk_budget"


def test_contributions_match_the_budgets() -> None:
    df = _returns()
    budgets = {"aaa": 0.5, "bbb": 0.3, "ccc": 0.2}
    weights = risk_budget_weights(df, budgets)
    # risk_contributions returns fractional variance shares summing to 1
    shares = risk_contributions(weights.to_numpy(), df.cov().to_numpy())
    assert shares[0] == pytest.approx(0.5, abs=1e-4)
    assert shares[1] == pytest.approx(0.3, abs=1e-4)
    assert shares[2] == pytest.approx(0.2, abs=1e-4)


def test_weights_sum_to_one_and_stay_positive() -> None:
    df = _returns()
    weights = risk_budget_weights(df, {"aaa": 0.7, "bbb": 0.2, "ccc": 0.1})
    assert float(weights.sum()) == pytest.approx(1.0)
    assert (weights.to_numpy() > 0).all()


def test_budget_scale_is_irrelevant() -> None:
    df = _returns()
    a = risk_budget_weights(df, {"aaa": 5.0, "bbb": 3.0, "ccc": 2.0})
    b = risk_budget_weights(df, {"aaa": 0.5, "bbb": 0.3, "ccc": 0.2})
    assert np.allclose(a.to_numpy(), b.to_numpy())


def test_bigger_budget_means_bigger_weight_for_iid_assets() -> None:
    rng = np.random.default_rng(9)
    df = pd.DataFrame(rng.normal(0, 0.01, size=(500, 2)), columns=["x", "y"])
    weights = risk_budget_weights(df, {"x": 0.8, "y": 0.2})
    assert weights["x"] > weights["y"]


def test_accepts_explicit_covariance() -> None:
    df = _returns()
    weights = risk_budget_weights(df, {"aaa": 1.0, "bbb": 1.0, "ccc": 1.0}, cov=df.cov())
    assert float(weights.sum()) == pytest.approx(1.0)


def test_bad_budgets_raise() -> None:
    df = _returns()
    with pytest.raises(ValueError, match="missing tickers"):
        risk_budget_weights(df, {"aaa": 1.0, "bbb": 1.0})
    with pytest.raises(ValueError, match="> 0"):
        risk_budget_weights(df, {"aaa": 1.0, "bbb": 1.0, "ccc": 0.0})
    with pytest.raises(ValueError, match="finite"):
        risk_budget_weights(df, {"aaa": 1.0, "bbb": 1.0, "ccc": float("nan")})
