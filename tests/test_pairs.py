"""Tests for the pairs-trading / cointegration module."""

import numpy as np
import pandas as pd
import pytest

from src.strategy.pairs import (
    CointegrationResult,
    engle_granger_test,
    pairs_trading_signal,
)


def _cointegrated_pair(n: int = 500, seed: int = 0):
    """Two series sharing a common stochastic trend + mean-reverting spread.

    The common factor std (0.05) dominates the spread std (~0.07) so the
    OLS regression cleanly recovers the true hedge ratio of 1.5.
    """
    rng = np.random.default_rng(seed)
    common = np.cumsum(rng.normal(0, 0.05, n))
    # spread is an AR(1) with phi=0.7 — strongly mean-reverting
    spread = np.zeros(n)
    for t in range(1, n):
        spread[t] = 0.7 * spread[t - 1] + rng.normal(0, 0.05)
    x = 50 + common
    y = 50 + 1.5 * common + spread  # true hedge ratio 1.5
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    return pd.Series(y, index=idx, name="Y"), pd.Series(x, index=idx, name="X")


def _independent_pair(n: int = 500, seed: int = 0):
    """Two random walks with no common factor → not cointegrated."""
    rng = np.random.default_rng(seed)
    y = 50 + np.cumsum(rng.normal(0, 0.5, n))
    x = 50 + np.cumsum(rng.normal(0, 0.5, n))
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    return pd.Series(y, index=idx, name="Y"), pd.Series(x, index=idx, name="X")


# ---------------------------------------------------------------------------
# engle_granger_test
# ---------------------------------------------------------------------------


def test_engle_granger_returns_dataclass():
    y, x = _cointegrated_pair()
    res = engle_granger_test(y, x)
    assert isinstance(res, CointegrationResult)
    assert isinstance(res.residuals, pd.Series)
    assert len(res.residuals) == len(y)


def test_engle_granger_detects_cointegrated_pair():
    y, x = _cointegrated_pair(n=1000, seed=42)
    res = engle_granger_test(y, x)
    assert res.is_cointegrated
    # true hedge ratio is 1.5 — OLS should recover it within ~10%
    assert res.hedge_ratio == pytest.approx(1.5, rel=0.1)


def test_engle_granger_rejects_independent_walks():
    y, x = _independent_pair(n=500, seed=7)
    res = engle_granger_test(y, x)
    assert not res.is_cointegrated


def test_engle_granger_raises_on_short_series():
    idx = pd.date_range("2020-01-01", periods=5, freq="B")
    y = pd.Series([1, 2, 3, 4, 5], index=idx)
    x = pd.Series([2, 3, 4, 5, 6], index=idx)
    with pytest.raises(ValueError, match="observations"):
        engle_granger_test(y, x)


# ---------------------------------------------------------------------------
# pairs_trading_signal
# ---------------------------------------------------------------------------


def test_pairs_signal_returns_engine_compatible_frame():
    y, x = _cointegrated_pair(n=800, seed=1)
    out = pairs_trading_signal(y, x)
    assert {"close", "spread_z", "signal"} <= set(out.columns)
    assert out["signal"].isin([-1, 0, 1]).all()
    # metadata published via .attrs
    assert "hedge_ratio" in out.attrs


def test_pairs_signal_generates_trades_on_cointegrated_pair():
    y, x = _cointegrated_pair(n=800, seed=2)
    out = pairs_trading_signal(y, x, z_window=30, z_entry=1.5, z_exit=0.3)
    # at least some non-zero positions
    assert (out["signal"] != 0).any()
    # entry/exit z-score thresholds are respected at boundaries
    long_entries = (out["signal"].diff() > 0) & (out["signal"] == 1)
    if long_entries.any():
        first = out.index[long_entries][0]
        assert out.loc[first, "spread_z"] <= -1.5 + 1e-9


def test_pairs_signal_requires_cointegration_by_default():
    y, x = _independent_pair(n=500, seed=8)
    with pytest.raises(ValueError, match="not cointegrated"):
        pairs_trading_signal(y, x)


def test_pairs_signal_can_skip_cointegration_check():
    y, x = _independent_pair(n=500, seed=9)
    # must not raise
    out = pairs_trading_signal(y, x, require_cointegration=False)
    assert "signal" in out.columns
