"""Tests for portfolio risk / diversification analytics."""

import pytest

from src.portfolio.analytics import (
    diversification_ratio,
    effective_number_of_assets,
    portfolio_volatility,
    risk_contributions,
)

# Two uncorrelated assets with vols 0.2 and 0.3.
_COV = [[0.04, 0.0], [0.0, 0.09]]
_EQUAL = [0.5, 0.5]


def test_portfolio_volatility_uncorrelated() -> None:
    # var = 0.25*0.04 + 0.25*0.09 = 0.0325
    assert portfolio_volatility(_EQUAL, _COV) == pytest.approx(0.0325**0.5)


def test_portfolio_volatility_single_asset() -> None:
    assert portfolio_volatility([1.0], [[0.04]]) == pytest.approx(0.2)


def test_risk_contributions_sum_to_one() -> None:
    rc = risk_contributions([0.3, 0.7], _COV)
    assert rc.sum() == pytest.approx(1.0)


def test_risk_contributions_equal_for_symmetric_case() -> None:
    # equal weights + equal vols + uncorrelated -> equal risk contributions
    rc = risk_contributions(_EQUAL, [[0.04, 0.0], [0.0, 0.04]])
    assert rc == pytest.approx([0.5, 0.5])


def test_risk_contributions_zero_variance() -> None:
    rc = risk_contributions([0.5, 0.5], [[0.0, 0.0], [0.0, 0.0]])
    assert (rc == 0).all()


def test_diversification_ratio_uncorrelated_equal() -> None:
    # equal-weight, equal-vol, uncorrelated -> DR = sqrt(2)
    dr = diversification_ratio(_EQUAL, [[0.04, 0.0], [0.0, 0.04]])
    assert dr == pytest.approx(2**0.5)


def test_diversification_ratio_single_asset_is_one() -> None:
    assert diversification_ratio([1.0], [[0.09]]) == pytest.approx(1.0)


def test_effective_number_of_assets() -> None:
    assert effective_number_of_assets([0.25, 0.25, 0.25, 0.25]) == pytest.approx(4.0)
    assert effective_number_of_assets([1.0, 0.0, 0.0]) == pytest.approx(1.0)
    assert effective_number_of_assets([0.0, 0.0]) == 0.0


def test_shape_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="does not match"):
        portfolio_volatility([0.5, 0.5], [[0.04]])
