"""Tests for intraday volume-profile generators."""

import numpy as np
import pytest

from src.execution import intraday_volume_profile, vwap_schedule


def test_weights_sum_to_one() -> None:
    for n in (1, 5, 13, 78):
        w = intraday_volume_profile(n)
        assert w.shape == (n,)
        assert float(w.sum()) == pytest.approx(1.0)
        assert (w >= 0).all()


def test_u_shape_elevates_the_ends() -> None:
    w = intraday_volume_profile(13, shape="u", depth=0.4)
    mid = len(w) // 2
    assert w[0] > w[mid]  # open busier than midday
    assert w[-1] > w[mid]  # close busier than midday
    assert w[0] == pytest.approx(w[-1])  # symmetric


def test_midday_is_the_trough() -> None:
    w = intraday_volume_profile(21, shape="u")
    assert int(np.argmin(w)) == len(w) // 2  # the minimum sits at the middle


def test_deeper_depth_digs_a_bigger_lull() -> None:
    shallow = intraday_volume_profile(21, shape="u", depth=0.8)
    deep = intraday_volume_profile(21, shape="u", depth=0.2)
    mid = 10
    # a smaller depth makes the midday bucket a smaller share of the day
    assert deep[mid] < shallow[mid]
    # ...and the ends a larger share
    assert deep[0] > shallow[0]


def test_depth_one_is_flat() -> None:
    w = intraday_volume_profile(10, shape="u", depth=1.0)
    assert np.allclose(w, 1.0 / 10)


def test_flat_shape_is_uniform() -> None:
    w = intraday_volume_profile(8, shape="flat")
    assert np.allclose(w, 1.0 / 8)


def test_single_bucket() -> None:
    assert np.array_equal(intraday_volume_profile(1), [1.0])


def test_feeds_the_vwap_scheduler() -> None:
    profile = intraday_volume_profile(20, shape="u")
    schedule = vwap_schedule(10_000.0, profile)
    assert float(np.sum(schedule)) == pytest.approx(10_000.0)
    # the VWAP order trades more at the liquid open than the quiet midday
    assert schedule[0] > schedule[10]


def test_bad_inputs_raise() -> None:
    with pytest.raises(ValueError, match="n_buckets"):
        intraday_volume_profile(0)
    with pytest.raises(ValueError, match="shape"):
        intraday_volume_profile(10, shape="w")
    with pytest.raises(ValueError, match="depth"):
        intraday_volume_profile(10, shape="u", depth=0.0)
    with pytest.raises(ValueError, match="depth"):
        intraday_volume_profile(10, shape="u", depth=1.5)
