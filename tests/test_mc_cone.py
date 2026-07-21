"""Tests for the bootstrap equity confidence cone."""

import numpy as np
import pandas as pd
import pytest

from src.reporting.mc_cone import equity_cone


def _returns(n: int = 250, mean: float = 0.0005, vol: float = 0.01, seed: int = 1) -> pd.Series:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2022-01-03", periods=n, freq="B")
    return pd.Series(rng.normal(mean, vol, n), index=idx)


def test_shape_columns_and_index() -> None:
    r = _returns()
    cone = equity_cone(r, n_simulations=500, percentiles=(5, 25, 50, 75, 95))
    assert list(cone.columns) == ["p5", "p25", "p50", "p75", "p95"]
    assert cone.index.equals(r.index)
    assert len(cone) == len(r)


def test_percentiles_are_ordered_at_every_bar() -> None:
    cone = equity_cone(_returns(), n_simulations=500)
    values = cone.to_numpy()
    # each row must be non-decreasing across the sorted percentile columns
    assert (np.diff(values, axis=1) >= -1e-9).all()


def test_cone_anchored_and_widens_with_horizon() -> None:
    cone = equity_cone(_returns(), n_simulations=800)
    # every band starts at the initial level
    assert np.allclose(cone.iloc[0].to_numpy(), 1.0)
    early_spread = cone["p95"].iloc[10] - cone["p5"].iloc[10]
    late_spread = cone["p95"].iloc[-1] - cone["p5"].iloc[-1]
    assert late_spread > early_spread  # the cone fans out


def test_median_tracks_the_compounded_mean() -> None:
    # with a clear positive drift the median path ends above the start
    r = _returns(mean=0.001, vol=0.005, n=300)
    cone = equity_cone(r, n_simulations=1000)
    assert cone["p50"].iloc[-1] > 1.0
    assert cone["p5"].iloc[-1] < cone["p50"].iloc[-1] < cone["p95"].iloc[-1]


def test_custom_initial_scales_the_cone() -> None:
    r = _returns()
    base = equity_cone(r, n_simulations=400, initial=1.0, seed=7)
    scaled = equity_cone(r, n_simulations=400, initial=100_000.0, seed=7)
    assert np.allclose(scaled.to_numpy(), base.to_numpy() * 100_000.0)


def test_reproducible_with_seed() -> None:
    r = _returns()
    a = equity_cone(r, n_simulations=300, seed=11)
    b = equity_cone(r, n_simulations=300, seed=11)
    assert np.array_equal(a.to_numpy(), b.to_numpy())


def test_block_bootstrap_runs() -> None:
    cone = equity_cone(_returns(), n_simulations=300, block_size=10)
    assert np.isfinite(cone.to_numpy()).all()
    assert np.allclose(cone.iloc[0].to_numpy(), 1.0)


def test_single_percentile_is_a_valid_path() -> None:
    cone = equity_cone(_returns(), n_simulations=300, percentiles=(50.0,))
    assert list(cone.columns) == ["p50"]
    assert (cone["p50"] > 0).all()


def test_bad_inputs_raise() -> None:
    r = _returns(50)
    with pytest.raises(ValueError, match="empty"):
        equity_cone(pd.Series(dtype=float))
    with pytest.raises(ValueError, match="NaN"):
        bad = r.copy()
        bad.iloc[2] = np.nan
        equity_cone(bad)
    with pytest.raises(ValueError, match="n_simulations"):
        equity_cone(r, n_simulations=0)
    with pytest.raises(ValueError, match="block_size"):
        equity_cone(r, block_size=51)
    with pytest.raises(ValueError, match="initial"):
        equity_cone(r, initial=0.0)
    with pytest.raises(ValueError, match="percentile"):
        equity_cone(r, percentiles=(0.0, 50.0))
