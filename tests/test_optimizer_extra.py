"""Tests for the added portfolio optimisers (max diversification, HRP)."""

import numpy as np
import pandas as pd
import pytest

from src.portfolio.optimizer import (
    hierarchical_risk_parity_weights,
    maximum_diversification_weights,
)


def _frame(cols: list[str]) -> pd.DataFrame:
    """Dummy returns frame — only its columns are used when cov is supplied."""
    return pd.DataFrame(np.zeros((5, len(cols))), columns=cols)


def _cov(matrix: list[list[float]], cols: list[str]) -> pd.DataFrame:
    return pd.DataFrame(matrix, index=cols, columns=cols)


# --- maximum diversification ----------------------------------------------


def test_mdp_uncorrelated_is_inverse_vol() -> None:
    cols = ["A", "B"]
    cov = _cov([[0.01, 0.0], [0.0, 0.04]], cols)  # vols 0.1, 0.2
    w = maximum_diversification_weights(_frame(cols), cov=cov)
    assert w["A"] == pytest.approx(2 / 3, abs=1e-3)
    assert w["B"] == pytest.approx(1 / 3, abs=1e-3)


def test_mdp_weights_valid_on_random() -> None:
    rng = np.random.default_rng(0)
    cols = list("ABCD")
    rets = pd.DataFrame(rng.normal(0, 0.01, (400, 4)), columns=cols)
    w = maximum_diversification_weights(rets)
    assert w.sum() == pytest.approx(1.0)
    assert (w >= 0).all()


# --- hierarchical risk parity ---------------------------------------------


def test_hrp_two_assets_is_inverse_variance() -> None:
    cols = ["A", "B"]
    cov = _cov([[0.01, 0.0], [0.0, 0.04]], cols)  # var ratio 1:4 -> weights 0.8:0.2
    w = hierarchical_risk_parity_weights(_frame(cols), cov=cov)
    assert w["A"] == pytest.approx(0.8, abs=1e-3)
    assert w["B"] == pytest.approx(0.2, abs=1e-3)


def test_hrp_four_uncorrelated_equal_vol_is_equal_weight() -> None:
    cols = list("ABCD")
    cov = _cov((0.04 * np.eye(4)).tolist(), cols)
    w = hierarchical_risk_parity_weights(_frame(cols), cov=cov)
    assert np.allclose(w.to_numpy(), 0.25, atol=1e-6)


def test_hrp_weights_valid_on_random() -> None:
    rng = np.random.default_rng(1)
    cols = list("ABCDE")
    rets = pd.DataFrame(rng.normal(0, 0.01, (500, 5)), columns=cols)
    w = hierarchical_risk_parity_weights(rets)
    assert w.sum() == pytest.approx(1.0)
    assert (w >= 0).all()


def test_hrp_is_deterministic() -> None:
    rng = np.random.default_rng(2)
    cols = list("ABCD")
    rets = pd.DataFrame(rng.normal(0, 0.01, (300, 4)), columns=cols)
    a = hierarchical_risk_parity_weights(rets)
    b = hierarchical_risk_parity_weights(rets)
    assert np.allclose(a.to_numpy(), b.to_numpy())


def test_hrp_single_asset() -> None:
    cols = ["A"]
    w = hierarchical_risk_parity_weights(_frame(cols), cov=_cov([[0.01]], cols))
    assert w["A"] == pytest.approx(1.0)
