"""Tests for the return-series transform toolkit."""

import numpy as np
import pandas as pd
import pytest

from src.data.returns import (
    excess_returns,
    log_returns,
    log_to_simple,
    returns_to_prices,
    simple_returns,
    simple_to_log,
)


def _prices(n: int = 100, seed: int = 4) -> pd.Series:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=n, freq="B")
    return pd.Series(100.0 * np.cumprod(1 + rng.normal(0.0005, 0.01, n)), index=idx, name="px")


def test_simple_returns_values_and_warm_up() -> None:
    prices = pd.Series([100.0, 110.0, 99.0])
    out = simple_returns(prices)
    assert np.isnan(out.iloc[0])
    assert out.iloc[1] == pytest.approx(0.10)
    assert out.iloc[2] == pytest.approx(-0.10)


def test_log_returns_match_log_of_ratio() -> None:
    prices = _prices()
    out = log_returns(prices)
    expected = np.log(prices.to_numpy()[1:] / prices.to_numpy()[:-1])
    assert np.allclose(out.to_numpy()[1:], expected)


def test_round_trip_prices_to_returns_and_back() -> None:
    prices = _prices()
    r = simple_returns(prices)
    rebuilt = returns_to_prices(r.dropna(), initial=float(prices.iloc[0]))
    assert np.allclose(rebuilt.to_numpy(), prices.to_numpy()[1:])


def test_simple_log_conversions_are_inverses() -> None:
    r = simple_returns(_prices()).dropna()
    assert np.allclose(log_to_simple(simple_to_log(r)).to_numpy(), r.to_numpy())
    # and simple_to_log(simple_returns) equals log_returns
    prices = _prices()
    assert np.allclose(
        simple_to_log(simple_returns(prices)).to_numpy()[1:],
        log_returns(prices).to_numpy()[1:],
    )


def test_dataframe_support_preserves_labels() -> None:
    prices = pd.DataFrame({"a": [100.0, 110.0, 121.0], "b": [50.0, 45.0, 54.0]})
    out = simple_returns(prices)
    assert isinstance(out, pd.DataFrame)
    assert list(out.columns) == ["a", "b"]
    assert out.iloc[1, 0] == pytest.approx(0.10)
    assert out.iloc[1, 1] == pytest.approx(-0.10)
    rebuilt = returns_to_prices(out.iloc[1:], initial=1.0)
    assert rebuilt.shape == (2, 2)


def test_excess_returns_with_annual_scalar() -> None:
    idx = pd.date_range("2024-01-01", periods=4, freq="B")
    returns = pd.Series([0.001, 0.002, -0.001, 0.0], index=idx)
    out = excess_returns(returns, risk_free=0.05, periods_per_year=252)
    per_bar = (1.05) ** (1 / 252) - 1
    assert np.allclose(out.to_numpy(), returns.to_numpy() - per_bar)


def test_excess_returns_with_series_and_frame_broadcast() -> None:
    idx = pd.date_range("2024-01-01", periods=3, freq="B")
    rf = pd.Series([0.0001, 0.0002, 0.0001], index=idx)
    frame = pd.DataFrame({"a": [0.01, 0.02, 0.03], "b": [0.0, 0.0, 0.0]}, index=idx)
    out = excess_returns(frame, risk_free=rf)
    assert out.loc[idx[1], "a"] == pytest.approx(0.02 - 0.0002)
    assert out.loc[idx[1], "b"] == pytest.approx(-0.0002)


def test_zero_risk_free_is_identity() -> None:
    returns = simple_returns(_prices()).dropna()
    out = excess_returns(returns, risk_free=0.0)
    assert np.allclose(out.to_numpy(), returns.to_numpy())


def test_invalid_prices_raise() -> None:
    with pytest.raises(ValueError, match="positive"):
        simple_returns(pd.Series([100.0, -1.0]))
    with pytest.raises(ValueError, match="positive"):
        log_returns(pd.Series([100.0, np.nan]))


def test_invalid_returns_and_params_raise() -> None:
    idx = pd.date_range("2024-01-01", periods=3, freq="B")
    returns = pd.Series([0.01, np.nan, 0.02], index=idx)
    with pytest.raises(ValueError, match="NaN"):
        returns_to_prices(returns)
    with pytest.raises(ValueError, match="initial"):
        returns_to_prices(returns.fillna(0.0), initial=0.0)
    clean = returns.fillna(0.0)
    with pytest.raises(ValueError, match="index"):
        excess_returns(clean, risk_free=pd.Series([0.0], index=idx[:1]))
    with pytest.raises(ValueError, match="risk_free"):
        excess_returns(clean, risk_free=-1.5)
    with pytest.raises(ValueError, match="periods_per_year"):
        excess_returns(clean, risk_free=0.05, periods_per_year=0)
