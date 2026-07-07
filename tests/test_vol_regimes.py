"""Tests for volatility-regime classification with hysteresis."""

import numpy as np
import pandas as pd
import pytest

from src.regime import VolRegime, realized_volatility, vol_regimes
from src.regime.volatility import _run_state_machine


def _regime_shift_returns(n_calm: int = 300, n_wild: int = 150, seed: int = 0) -> pd.Series:
    rng = np.random.default_rng(seed)
    calm = rng.normal(0.0, 0.004, n_calm)
    wild = rng.normal(0.0, 0.030, n_wild)
    idx = pd.date_range("2020-01-01", periods=n_calm + n_wild, freq="B")
    return pd.Series(np.concatenate([calm, wild]), index=idx)


def test_realized_volatility_annualises() -> None:
    # alternating +1%/-1% has per-bar std ~0.01 -> annualised ~0.1587
    returns = pd.Series([0.01, -0.01] * 50)
    vol = realized_volatility(returns, window=20)
    assert vol.name == "realized_vol"
    assert vol.iloc[:19].isna().all()
    expected = pd.Series([0.01, -0.01] * 10).std() * np.sqrt(252)
    assert vol.iloc[-1] == pytest.approx(expected)


def test_codes_are_valid_regimes() -> None:
    regimes = vol_regimes(_regime_shift_returns())
    assert regimes.name == "vol_regime"
    assert set(np.unique(regimes.to_numpy())).issubset({0, 1, 2})


def test_vol_spike_enters_high_regime() -> None:
    returns = _regime_shift_returns(n_calm=300, n_wild=150)
    regimes = vol_regimes(returns, window=20, lookback=252)
    # well after the volatility jump the classifier must sit in HIGH
    post_jump = regimes.iloc[330:360]
    assert (post_jump == int(VolRegime.HIGH)).mean() > 0.9
    # the thresholds are relative (trailing quantiles), so even the calm
    # stationary stretch spends *some* time flagged HIGH — but far less
    # than the genuinely wild segment
    calm_share = float((regimes.iloc[100:290] == int(VolRegime.HIGH)).mean())
    assert calm_share < 0.5


def test_warm_up_is_normal() -> None:
    regimes = vol_regimes(_regime_shift_returns(), window=20)
    assert (regimes.iloc[:19] == int(VolRegime.NORMAL)).all()


def test_hysteresis_state_machine_exact_path() -> None:
    # constant thresholds: low=0.5, mid=1.5, high=2.0
    n = 7
    vol = np.array([1.0, 2.5, 1.8, 1.4, 0.4, 1.0, 1.6])
    lo = np.full(n, 0.5)
    mid = np.full(n, 1.5)
    hi = np.full(n, 2.0)
    states = _run_state_machine(vol, lo, mid, hi)
    # 1.8 stays HIGH (above mid), 1.4 demotes; 1.0 stays LOW (below mid),
    # 1.6 promotes back to NORMAL
    assert list(states) == [1, 2, 2, 1, 0, 0, 1]


def test_nan_inputs_keep_previous_state() -> None:
    vol = np.array([2.5, np.nan, 1.0])
    lo = np.full(3, 0.5)
    mid = np.full(3, 1.5)
    hi = np.full(3, 2.0)
    states = _run_state_machine(vol, lo, mid, hi)
    assert list(states) == [2, 2, 1]


def test_bad_windows_raise() -> None:
    returns = _regime_shift_returns()
    with pytest.raises(ValueError, match="window"):
        vol_regimes(returns, window=1)
    with pytest.raises(ValueError, match="lookback"):
        vol_regimes(returns, window=20, lookback=10)


def test_bad_quantiles_raise() -> None:
    returns = _regime_shift_returns()
    with pytest.raises(ValueError, match="quantile"):
        vol_regimes(returns, low_quantile=0.8, high_quantile=0.2)
    with pytest.raises(ValueError, match="quantile"):
        vol_regimes(returns, low_quantile=0.0)
