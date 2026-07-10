"""Tests for regime-conditional performance statistics."""

import numpy as np
import pandas as pd
import pytest

from src.regime import regime_performance, vol_regimes


def _idx(n: int) -> pd.DatetimeIndex:
    return pd.date_range("2024-01-01", periods=n, freq="B")


def _two_regime_sample() -> tuple[pd.Series, pd.Series]:
    # regime 0: 60 bars of +1%; regime 1: 40 bars of -0.5%
    idx = _idx(100)
    returns = pd.Series(np.where(np.arange(100) < 60, 0.01, -0.005), index=idx)
    regimes = pd.Series(np.where(np.arange(100) < 60, 0, 1), index=idx)
    return returns, regimes


def test_per_regime_means_and_shares() -> None:
    returns, regimes = _two_regime_sample()
    table = regime_performance(returns, regimes)
    assert list(table.index) == [0, 1]
    assert table.loc[0, "n_obs"] == 60
    assert table.loc[0, "share"] == pytest.approx(0.6)
    assert table.loc[0, "ann_return"] == pytest.approx(0.01 * 252)
    assert table.loc[1, "ann_return"] == pytest.approx(-0.005 * 252)
    assert table["share"].sum() == pytest.approx(1.0)


def test_hit_rate_and_extremes() -> None:
    returns, regimes = _two_regime_sample()
    table = regime_performance(returns, regimes)
    assert table.loc[0, "hit_rate"] == 1.0
    assert table.loc[1, "hit_rate"] == 0.0
    assert table.loc[0, "best"] == 0.01
    assert table.loc[0, "worst"] == 0.01
    assert table.loc[1, "worst"] == -0.005


def test_zero_volatility_regime_has_nan_sharpe() -> None:
    # binary-exact values (2^-6, -2^-5) so the within-regime mean is exact
    # and the std is exactly 0 — no float dust
    idx = _idx(8)
    returns = pd.Series([0.015625] * 4 + [-0.03125] * 4, index=idx)
    regimes = pd.Series([0] * 4 + [1] * 4, index=idx)
    table = regime_performance(returns, regimes)
    assert table.loc[0, "ann_vol"] == 0.0
    assert np.isnan(table.loc[0, "sharpe"])


def test_vol_and_sharpe_match_direct_computation() -> None:
    rng = np.random.default_rng(9)
    idx = _idx(200)
    returns = pd.Series(rng.normal(0.001, 0.01, 200), index=idx)
    regimes = pd.Series([0] * 120 + [1] * 80, index=idx)
    table = regime_performance(returns, regimes)
    sub = returns.iloc[:120]
    expected_vol = float(sub.std(ddof=1)) * np.sqrt(252)
    assert table.loc[0, "ann_vol"] == pytest.approx(expected_vol)
    assert table.loc[0, "sharpe"] == pytest.approx(float(sub.mean()) * 252 / expected_vol)


def test_nan_labels_are_skipped_and_shares_shrink() -> None:
    returns, regimes = _two_regime_sample()
    regimes = regimes.astype(float)
    regimes.iloc[:10] = np.nan  # detector warm-up
    table = regime_performance(returns, regimes)
    assert table["n_obs"].sum() == 90
    assert table["share"].sum() == pytest.approx(0.9)


def test_single_bar_regime_has_nan_vol() -> None:
    idx = _idx(3)
    returns = pd.Series([0.01, 0.02, 0.03], index=idx)
    regimes = pd.Series([0, 0, 1], index=idx)
    table = regime_performance(returns, regimes)
    assert table.loc[1, "n_obs"] == 1
    assert np.isnan(table.loc[1, "ann_vol"])
    assert np.isnan(table.loc[1, "sharpe"])


def test_string_labels_sort_and_work() -> None:
    idx = _idx(4)
    returns = pd.Series([0.01, -0.01, 0.02, -0.02], index=idx)
    regimes = pd.Series(["up", "down", "up", "down"], index=idx)
    table = regime_performance(returns, regimes)
    assert list(table.index) == ["down", "up"]
    assert table.loc["up", "hit_rate"] == 1.0


def test_integrates_with_vol_regimes() -> None:
    rng = np.random.default_rng(0)
    idx = _idx(450)
    returns = pd.Series(
        np.concatenate([rng.normal(0, 0.004, 300), rng.normal(0, 0.03, 150)]), index=idx
    )
    table = regime_performance(returns, vol_regimes(returns))
    # the HIGH regime (code 2) must show markedly higher realised vol
    assert table.loc[2, "ann_vol"] > 2 * table.loc[1, "ann_vol"]
    assert table["n_obs"].sum() == 450


def test_bad_inputs_raise() -> None:
    returns, regimes = _two_regime_sample()
    with pytest.raises(ValueError, match="index"):
        regime_performance(returns.iloc[:-1], regimes)
    with pytest.raises(ValueError, match="empty"):
        empty = pd.Series(dtype=float, index=pd.DatetimeIndex([]))
        regime_performance(empty, empty)
    with pytest.raises(ValueError, match="periods_per_year"):
        regime_performance(returns, regimes, periods_per_year=0)
    with pytest.raises(ValueError, match="non-NaN"):
        regime_performance(returns, pd.Series(np.nan, index=returns.index))
