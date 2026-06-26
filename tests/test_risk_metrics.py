"""Tests for the advanced risk-metrics module."""

import numpy as np
import pandas as pd
import pytest

from src.risk.metrics import (
    DrawdownStats,
    common_ratio,
    downside_deviation,
    drawdown_stats,
    gain_to_pain_ratio,
    historical_cvar,
    historical_var,
    omega_ratio,
    parametric_var,
    rolling_beta,
    tail_ratio,
    ulcer_index,
    upside_deviation,
)


@pytest.fixture
def daily_returns():
    rng = np.random.default_rng(0)
    dates = pd.date_range("2020-01-01", periods=500, freq="B")
    return pd.Series(rng.normal(0.0005, 0.01, len(dates)), index=dates)


# ---------------------------------------------------------------------------
# VaR / CVaR
# ---------------------------------------------------------------------------


class TestVaR:
    def test_historical_var_positive_loss(self, daily_returns):
        v = historical_var(daily_returns, level=0.05)
        assert v >= 0

    def test_cvar_geq_var(self, daily_returns):
        v = historical_var(daily_returns)
        c = historical_cvar(daily_returns)
        assert c >= v

    def test_parametric_var_close_to_historical_for_normal_returns(self, daily_returns):
        # for normally distributed returns, parametric ~= historical
        v_h = historical_var(daily_returns, 0.05)
        v_p = parametric_var(daily_returns, 0.05)
        assert v_p == pytest.approx(v_h, rel=0.30)

    def test_var_rejects_bad_level(self):
        for fn in (historical_var, historical_cvar, parametric_var):
            with pytest.raises(ValueError):
                fn(pd.Series([0.01]), level=0.0)
            with pytest.raises(ValueError):
                fn(pd.Series([0.01]), level=1.0)

    def test_var_zero_on_no_losses(self):
        only_gains = pd.Series([0.01, 0.02, 0.03, 0.005, 0.01])
        assert historical_var(only_gains) == 0.0


# ---------------------------------------------------------------------------
# Omega / Ulcer / G2P
# ---------------------------------------------------------------------------


class TestRatios:
    def test_omega_ratio_balanced(self):
        # symmetric returns around 0 → omega ~= 1
        r = pd.Series([-0.01, 0.01, -0.02, 0.02, -0.005, 0.005])
        assert omega_ratio(r) == pytest.approx(1.0, abs=1e-9)

    def test_omega_gt_one_when_drift_positive(self):
        r = pd.Series([0.02, 0.01, -0.005, 0.015, -0.003])
        assert omega_ratio(r) > 1.0

    def test_ulcer_index_zero_for_monotone_up(self):
        r = pd.Series([0.01] * 50)
        assert ulcer_index(r) == pytest.approx(0.0)

    def test_ulcer_index_positive_on_drawdown(self, daily_returns):
        assert ulcer_index(daily_returns) > 0

    def test_gain_to_pain_balanced(self):
        r = pd.Series([0.01, -0.01, 0.02, -0.02])
        assert gain_to_pain_ratio(r) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Drawdown
# ---------------------------------------------------------------------------


class TestDrawdown:
    def test_drawdown_zero_on_monotone_up(self):
        r = pd.Series([0.01] * 100, index=pd.date_range("2020-01-01", periods=100, freq="D"))
        s = drawdown_stats(r)
        assert s.max_drawdown == 0.0
        assert s.duration_days == 0

    def test_drawdown_captures_recovery(self):
        # 100 -> 110 -> 90 -> 110 — drawdown of ~18% then recovery
        prices = [100, 110, 90, 100, 110]
        rets = pd.Series(prices, index=pd.date_range("2020-01-01", periods=5)).pct_change().dropna()
        s = drawdown_stats(rets)
        assert isinstance(s, DrawdownStats)
        assert s.max_drawdown < 0
        assert s.recovery_date is not None

    def test_drawdown_no_recovery_yet(self):
        # 100 -> 110 -> 90 (no recovery)
        prices = [100, 110, 90]
        rets = pd.Series(prices, index=pd.date_range("2020-01-01", periods=3)).pct_change().dropna()
        s = drawdown_stats(rets)
        assert s.recovery_date is None
        assert s.recovery_days is None


# ---------------------------------------------------------------------------
# Deviations / tail / common
# ---------------------------------------------------------------------------


class TestDeviationsAndTail:
    def test_downside_deviation_only_uses_negatives(self):
        r = pd.Series([0.05, 0.04, -0.01, 0.03, -0.02])
        d = downside_deviation(r)
        # std of [-0.01, -0.02] annualised
        manual = pd.Series([-0.01, -0.02]).std(ddof=1) * np.sqrt(252)
        assert d == pytest.approx(manual)

    def test_upside_deviation_only_uses_positives(self):
        r = pd.Series([0.05, 0.04, -0.01, 0.03, -0.02])
        u = upside_deviation(r)
        manual = pd.Series([0.05, 0.04, 0.03]).std(ddof=1) * np.sqrt(252)
        assert u == pytest.approx(manual)

    def test_tail_ratio_one_for_symmetric(self, daily_returns):
        # symmetric normal returns → tail ratio ~= 1
        rng = np.random.default_rng(1)
        sym = pd.Series(rng.normal(0, 0.01, 5000))
        assert tail_ratio(sym) == pytest.approx(1.0, rel=0.1)

    def test_common_ratio_positive_for_positive_drift(self, daily_returns):
        # daily_returns has 0.0005 mean drift → positive common ratio
        assert common_ratio(daily_returns) > 0


# ---------------------------------------------------------------------------
# Rolling beta
# ---------------------------------------------------------------------------


class TestRollingBeta:
    def test_rolling_beta_of_self_equals_one(self, daily_returns):
        beta = rolling_beta(daily_returns, daily_returns, window=60).dropna()
        assert beta.iloc[-1] == pytest.approx(1.0, abs=1e-9)

    def test_rolling_beta_shape(self, daily_returns):
        bench = daily_returns + 0.0001
        beta = rolling_beta(daily_returns, bench, window=60)
        assert len(beta) == len(daily_returns)
        assert beta.iloc[:59].isna().all()
