"""Tests for iceberg clip scheduling."""

import numpy as np
import pytest

from src.execution import iceberg_schedule


def test_exact_division_gives_equal_clips() -> None:
    clips = iceberg_schedule(1_000.0, display_size=250.0)
    assert np.allclose(clips, [250.0, 250.0, 250.0, 250.0])


def test_final_clip_absorbs_the_remainder() -> None:
    clips = iceberg_schedule(1_100.0, display_size=250.0)
    assert np.allclose(clips[:-1], 250.0)
    assert clips[-1] == pytest.approx(100.0)
    assert float(clips.sum()) == pytest.approx(1_100.0)


def test_small_parent_is_a_single_clip() -> None:
    clips = iceberg_schedule(75.0, display_size=250.0)
    assert list(clips) == [75.0]


def test_zero_parent_is_empty() -> None:
    assert len(iceberg_schedule(0.0, display_size=100.0)) == 0


def test_jitter_randomises_within_bounds() -> None:
    clips = iceberg_schedule(10_000.0, display_size=100.0, jitter=0.25, seed=7)
    body = clips[:-1]  # the final clip may be a partial remainder
    assert (body >= 100.0 * 0.75 - 1e-9).all()
    assert (body <= 100.0 * 1.25 + 1e-9).all()
    assert len(np.unique(np.round(body, 6))) > 1  # actually randomised
    assert float(clips.sum()) == pytest.approx(10_000.0)


def test_jittered_schedule_is_reproducible_with_seed() -> None:
    a = iceberg_schedule(5_000.0, 100.0, jitter=0.2, seed=42)
    b = iceberg_schedule(5_000.0, 100.0, jitter=0.2, seed=42)
    assert np.array_equal(a, b)
    c = iceberg_schedule(5_000.0, 100.0, jitter=0.2, seed=43)
    assert not np.array_equal(a, c)


def test_all_clips_positive() -> None:
    clips = iceberg_schedule(1_234.5, display_size=99.9, jitter=0.3, seed=1)
    assert (clips > 0).all()


def test_bad_inputs_raise() -> None:
    with pytest.raises(ValueError, match="total_quantity"):
        iceberg_schedule(-1.0, 100.0)
    with pytest.raises(ValueError, match="display_size"):
        iceberg_schedule(100.0, 0.0)
    with pytest.raises(ValueError, match="jitter"):
        iceberg_schedule(100.0, 10.0, jitter=1.0)
    with pytest.raises(ValueError, match="jitter"):
        iceberg_schedule(100.0, 10.0, jitter=-0.1)
