"""Tests for TWAP / VWAP / POV child-order schedules."""

import numpy as np
import pandas as pd
import pytest

from src.execution import pov_schedule, twap_schedule, vwap_schedule

# --- TWAP -------------------------------------------------------------------


def test_twap_equal_slices_sum_to_total() -> None:
    slices = twap_schedule(1_000.0, 7)
    assert len(slices) == 7
    assert float(slices.sum()) == pytest.approx(1_000.0, abs=1e-9)
    assert np.allclose(slices, 1_000.0 / 7)


def test_twap_single_slice_takes_everything() -> None:
    assert list(twap_schedule(500.0, 1)) == [500.0]


def test_twap_zero_quantity_gives_zeros() -> None:
    assert (twap_schedule(0.0, 5) == 0.0).all()


def test_twap_bad_params_raise() -> None:
    with pytest.raises(ValueError, match="n_slices"):
        twap_schedule(100.0, 0)
    with pytest.raises(ValueError, match="total_quantity"):
        twap_schedule(-1.0, 5)


# --- VWAP -------------------------------------------------------------------


def test_vwap_proportional_to_profile() -> None:
    children = vwap_schedule(600.0, [1.0, 2.0, 3.0])
    assert np.allclose(children, [100.0, 200.0, 300.0])
    assert float(np.sum(children)) == pytest.approx(600.0)


def test_vwap_zero_bucket_gets_nothing() -> None:
    children = vwap_schedule(100.0, [0.0, 1.0])
    assert children[0] == 0.0
    assert children[1] == pytest.approx(100.0)


def test_vwap_series_preserves_index() -> None:
    idx = pd.date_range("2024-01-02 09:30", periods=3, freq="h")
    profile = pd.Series([100.0, 300.0, 100.0], index=idx)
    children = vwap_schedule(500.0, profile)
    assert isinstance(children, pd.Series)
    assert children.index.equals(idx)
    assert children.name == "child_quantity"
    assert children.iloc[1] == pytest.approx(300.0)


def test_vwap_invalid_profile_raises() -> None:
    with pytest.raises(ValueError, match="positive bucket"):
        vwap_schedule(100.0, [0.0, 0.0])
    with pytest.raises(ValueError, match="non-negative"):
        vwap_schedule(100.0, [1.0, -1.0])
    with pytest.raises(ValueError, match="non-empty"):
        vwap_schedule(100.0, [])


# --- POV --------------------------------------------------------------------


def test_pov_trades_participation_until_filled() -> None:
    children = pov_schedule(500.0, [1_000.0] * 10, participation=0.1)
    # 100 per bucket -> filled after 5 buckets
    assert np.allclose(children[:5], 100.0)
    assert (children[5:] == 0.0).all()
    assert float(np.sum(children)) == pytest.approx(500.0)


def test_pov_caps_the_final_partial_bucket() -> None:
    children = pov_schedule(250.0, [1_000.0] * 5, participation=0.1)
    assert np.allclose(children, [100.0, 100.0, 50.0, 0.0, 0.0])


def test_pov_may_finish_unfilled_in_thin_volume() -> None:
    children = pov_schedule(1_000.0, [100.0] * 4, participation=0.05)
    assert float(np.sum(children)) == pytest.approx(20.0)  # 5 per bucket
    assert float(np.sum(children)) < 1_000.0


def test_pov_series_preserves_index() -> None:
    idx = pd.date_range("2024-01-02 09:30", periods=4, freq="h")
    children = pov_schedule(50.0, pd.Series([100.0] * 4, index=idx), participation=0.2)
    assert isinstance(children, pd.Series)
    assert children.index.equals(idx)


def test_pov_bad_participation_raises() -> None:
    with pytest.raises(ValueError, match="participation"):
        pov_schedule(100.0, [1_000.0], participation=0.0)
    with pytest.raises(ValueError, match="participation"):
        pov_schedule(100.0, [1_000.0], participation=1.5)
