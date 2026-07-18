"""Tests for the FIFO queue-position fill model."""

import numpy as np
import pandas as pd
import pytest

from src.execution import simulate_queue_fill


def _volume(values: list[float]) -> pd.Series:
    idx = pd.date_range("2024-01-02 09:30", periods=len(values), freq="min")
    return pd.Series(values, index=idx)


def test_hand_walked_queue_burn_through() -> None:
    # 500 ahead, order 300; volume 200/200/300 at the level
    out = simulate_queue_fill(_volume([200.0, 200.0, 300.0]), 300.0, queue_ahead=500.0)
    assert list(out["queue_remaining"]) == [300.0, 100.0, 0.0]
    assert list(out["filled"]) == [0.0, 0.0, 200.0]  # bar 3: 100 eats queue, 200 fills
    assert list(out["cumulative_filled"]) == [0.0, 0.0, 200.0]
    assert not bool(out["complete"].iloc[-1])  # only 200 of 300 done


def test_front_of_queue_fills_immediately() -> None:
    out = simulate_queue_fill(_volume([150.0, 150.0]), 100.0, queue_ahead=0.0)
    assert out["filled"].iloc[0] == pytest.approx(100.0)
    assert bool(out["complete"].iloc[0])
    assert out["filled"].iloc[1] == 0.0  # nothing left to fill


def test_completion_bar_and_steady_state() -> None:
    out = simulate_queue_fill(_volume([100.0] * 6), 250.0, queue_ahead=100.0)
    # bar1 eats the queue; bars 2-4 fill 100+100+50
    assert list(out["cumulative_filled"]) == [0.0, 100.0, 200.0, 250.0, 250.0, 250.0]
    assert list(out["complete"]) == [False, False, False, True, True, True]
    first_complete = out.index[out["complete"]][0]
    assert first_complete == out.index[3]


def test_zero_volume_bars_change_nothing() -> None:
    out = simulate_queue_fill(_volume([0.0, 0.0, 50.0]), 40.0, queue_ahead=10.0)
    assert list(out["filled"]) == [0.0, 0.0, 40.0]
    assert out["queue_remaining"].iloc[-1] == 0.0


def test_queue_priority_delays_versus_no_queue() -> None:
    volume = _volume([100.0] * 5)
    front = simulate_queue_fill(volume, 200.0, queue_ahead=0.0)
    back = simulate_queue_fill(volume, 200.0, queue_ahead=300.0)
    front_done = int(np.argmax(front["complete"].to_numpy()))
    back_done = int(np.argmax(back["complete"].to_numpy()))
    assert front_done < back_done


def test_bad_inputs_raise() -> None:
    volume = _volume([100.0])
    with pytest.raises(ValueError, match="order_quantity"):
        simulate_queue_fill(volume, 0.0)
    with pytest.raises(ValueError, match="queue_ahead"):
        simulate_queue_fill(volume, 100.0, queue_ahead=-1.0)
    with pytest.raises(ValueError, match="non-negative"):
        simulate_queue_fill(_volume([-5.0]), 100.0)
    with pytest.raises(ValueError, match="NaN"):
        simulate_queue_fill(_volume([np.nan]), 100.0)
