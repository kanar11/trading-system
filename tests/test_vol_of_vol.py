"""Tests for the volatility-of-volatility gauge."""

import numpy as np
import pandas as pd
import pytest

from src.regime import vol_of_vol


def _regime_shift(seed: int = 0) -> pd.Series:
    rng = np.random.default_rng(seed)
    calm = rng.normal(0.0, 0.004, 300)
    wild = rng.normal(0.0, 0.03, 150)
    return pd.Series(np.concatenate([calm, wild]))


def test_non_negative_and_named() -> None:
    vov = vol_of_vol(_regime_shift(), vol_window=20, vov_window=20)
    assert vov.name == "vol_of_vol"
    assert (vov.dropna() >= 0).all()


def test_spikes_at_the_regime_transition() -> None:
    vov = vol_of_vol(_regime_shift(), vol_window=20, vov_window=20)
    calm_level = float(vov.iloc[100:150].mean())  # deep in the calm regime
    transition = float(vov.iloc[295:340].max())  # around the vol jump
    assert transition > 10 * calm_level


def test_steady_vol_stays_low() -> None:
    rng = np.random.default_rng(1)
    steady = pd.Series(rng.normal(0.0, 0.01, 400))
    steady_vov = float(vol_of_vol(steady, 20, 20).dropna().mean())
    shift_peak = float(vol_of_vol(_regime_shift(seed=1), 20, 20).iloc[295:340].max())
    assert shift_peak > 5 * steady_vov


def test_warm_up_covers_both_windows() -> None:
    vov = vol_of_vol(_regime_shift(), vol_window=20, vov_window=20)
    # inner vol needs 20 bars, outer std another 20 -> first valid at 38
    assert vov.iloc[:38].isna().all()
    assert vov.iloc[38:].notna().all()


def test_relative_mode_is_unit_free_and_finite() -> None:
    r = _regime_shift()
    absolute = vol_of_vol(r, 20, 20, relative=False)
    relative = vol_of_vol(r, 20, 20, relative=True)
    assert np.isfinite(relative.dropna()).all()
    # a coefficient-of-variation is generally a different scale from the raw std
    assert not np.allclose(absolute.dropna().to_numpy(), relative.dropna().to_numpy())


def test_scaling_returns_leaves_relative_vov_unchanged() -> None:
    # doubling every return doubles vol and std-of-vol equally, so the
    # relative (CoV) vol-of-vol is scale-invariant
    r = _regime_shift(seed=2)
    a = vol_of_vol(r, 20, 20, relative=True).dropna()
    b = vol_of_vol(2.0 * r, 20, 20, relative=True).dropna()
    assert np.allclose(a.to_numpy(), b.to_numpy())


def test_bad_params_raise() -> None:
    r = _regime_shift()
    with pytest.raises(ValueError, match="vov_window"):
        vol_of_vol(r, vol_window=20, vov_window=1)
    with pytest.raises(ValueError, match="window"):
        vol_of_vol(r, vol_window=1, vov_window=20)
    with pytest.raises(ValueError, match="periods_per_year"):
        vol_of_vol(r, periods_per_year=0)
