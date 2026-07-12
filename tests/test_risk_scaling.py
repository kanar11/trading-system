"""Tests for constant-volatility (risk-managed) exposure scaling."""

import numpy as np
import pandas as pd
import pytest

from src.risk import apply_risk_scaling, risk_managed_scaling


def _switching_vol_returns(seed: int = 0) -> pd.Series:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2022-01-03", periods=500, freq="B")
    calm = rng.normal(0.0, 0.005, 250)
    wild = rng.normal(0.0, 0.025, 250)
    return pd.Series(np.concatenate([calm, wild]), index=idx)


def test_scale_matches_target_over_realized_vol() -> None:
    returns = _switching_vol_returns()
    scale = risk_managed_scaling(returns, target_vol=0.10, window=60, max_leverage=10.0)
    assert scale.name == "risk_scale"
    t = 200
    realized = float(returns.iloc[t - 59 : t + 1].std(ddof=1)) * np.sqrt(252)
    assert scale.iloc[t] == pytest.approx(0.10 / realized)


def test_high_vol_regime_gets_less_exposure() -> None:
    returns = _switching_vol_returns()
    scale = risk_managed_scaling(returns, target_vol=0.10, window=60, max_leverage=10.0)
    calm_scale = float(scale.iloc[150:250].mean())
    wild_scale = float(scale.iloc[350:].mean())
    assert wild_scale < calm_scale / 2


def test_max_leverage_caps_quiet_markets() -> None:
    returns = _switching_vol_returns()
    scale = risk_managed_scaling(returns, target_vol=0.40, window=60, max_leverage=2.0)
    assert float(scale.dropna().max()) <= 2.0
    # the calm half would demand > 2x without the cap, so the cap binds
    assert (scale.iloc[100:250] == 2.0).any()


def test_warm_up_is_nan() -> None:
    returns = _switching_vol_returns()
    scale = risk_managed_scaling(returns, window=60)
    assert scale.iloc[:59].isna().all()
    assert scale.iloc[60:].notna().all()


def test_scaled_series_vol_is_stabilised() -> None:
    # the point of Barroso & Santa-Clara: scaling flattens the vol regime shift
    returns = _switching_vol_returns()
    scale = risk_managed_scaling(returns, target_vol=0.10, window=60, max_leverage=10.0)
    scaled = apply_risk_scaling(returns, scale)
    raw_calm = float(returns.iloc[150:250].std())
    raw_wild = float(returns.iloc[350:].std())
    scaled_calm = float(scaled.iloc[150:250].std())
    scaled_wild = float(scaled.iloc[350:].std())
    assert raw_wild / raw_calm > 3  # the raw series has a big regime shift
    assert scaled_wild / scaled_calm < raw_wild / raw_calm / 2  # strongly damped


def test_apply_uses_previous_bar_decision() -> None:
    idx = pd.date_range("2024-01-01", periods=4, freq="B")
    returns = pd.Series([0.01, 0.02, -0.01, 0.03], index=idx)
    scale = pd.Series([1.0, 0.5, 2.0, 0.0], index=idx)
    out = apply_risk_scaling(returns, scale)
    assert out.iloc[0] == 0.0  # no prior decision -> no exposure
    assert out.iloc[1] == pytest.approx(0.02 * 1.0)
    assert out.iloc[2] == pytest.approx(-0.01 * 0.5)
    assert out.iloc[3] == pytest.approx(0.03 * 2.0)


def test_index_mismatch_raises() -> None:
    returns = _switching_vol_returns()
    scale = risk_managed_scaling(returns)
    with pytest.raises(ValueError, match="index"):
        apply_risk_scaling(returns.iloc[:-1], scale)


def test_bad_params_raise() -> None:
    returns = _switching_vol_returns()
    with pytest.raises(ValueError, match="target_vol"):
        risk_managed_scaling(returns, target_vol=0.0)
    with pytest.raises(ValueError, match="max_leverage"):
        risk_managed_scaling(returns, max_leverage=0.0)
    with pytest.raises(ValueError, match="window"):
        risk_managed_scaling(returns, window=1)
    with pytest.raises(ValueError, match="periods_per_year"):
        risk_managed_scaling(returns, periods_per_year=0)
