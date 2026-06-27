"""Tests for periodic-return analytics (annual / monthly tables, rolling)."""

import numpy as np
import pandas as pd
import pytest

from src.reporting.periodic import annual_returns, monthly_returns_table, rolling_metrics


def test_annual_returns_exact_small_case() -> None:
    r = pd.Series([0.1, 0.1], index=pd.to_datetime(["2020-01-02", "2020-01-03"]))
    out = annual_returns(r)
    assert out.loc[2020] == pytest.approx(1.1 * 1.1 - 1)


def test_annual_returns_splits_by_year() -> None:
    idx = pd.to_datetime(["2020-06-01", "2021-06-01"])
    out = annual_returns(pd.Series([0.05, -0.03], index=idx))
    assert list(out.index) == [2020, 2021]
    assert out.loc[2020] == pytest.approx(0.05)
    assert out.loc[2021] == pytest.approx(-0.03)


def test_annual_returns_empty() -> None:
    assert annual_returns(pd.Series([], dtype=float)).empty


def test_monthly_table_values_and_annual_total() -> None:
    jan = pd.Series([0.05, 0.05], index=pd.to_datetime(["2020-01-10", "2020-01-20"]))
    feb = pd.Series([0.02], index=pd.to_datetime(["2020-02-10"]))
    table = monthly_returns_table(pd.concat([jan, feb]))
    assert table.loc[2020, 1] == pytest.approx(0.1025)
    assert table.loc[2020, 2] == pytest.approx(0.02)
    assert table.loc[2020, "annual"] == pytest.approx(1.1025 * 1.02 - 1)


def test_monthly_table_empty() -> None:
    assert monthly_returns_table(pd.Series([], dtype=float)).empty


def test_rolling_metrics_columns_and_warmup() -> None:
    rng = np.random.default_rng(0)
    idx = pd.date_range("2020-01-01", periods=200, freq="B")
    r = pd.Series(rng.normal(0.0005, 0.01, 200), index=idx)
    out = rolling_metrics(r, window=20)
    assert list(out.columns) == ["return", "volatility", "sharpe"]
    assert out.iloc[:19].isna().all().all()
    assert out.iloc[-1].notna().all()


def test_rolling_metrics_volatility_positive() -> None:
    rng = np.random.default_rng(1)
    idx = pd.date_range("2020-01-01", periods=120, freq="B")
    r = pd.Series(rng.normal(0, 0.01, 120), index=idx)
    out = rolling_metrics(r, window=30).dropna()
    assert (out["volatility"] > 0).all()


def test_rolling_metrics_bad_window() -> None:
    with pytest.raises(ValueError, match="window"):
        rolling_metrics(pd.Series([0.01, 0.02]), window=1)
