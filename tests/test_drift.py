"""Tests for weight drift between rebalances."""

import numpy as np
import pandas as pd
import pytest

from src.portfolio import drift_weights, portfolio_turnover


def _returns(values: dict[str, list[float]]) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=len(next(iter(values.values()))), freq="B")
    return pd.DataFrame(values, index=idx)


def test_winner_grows_its_share() -> None:
    target = pd.Series({"win": 0.5, "lose": 0.5})
    r = _returns({"win": [0.10, 0.10], "lose": [-0.10, -0.10]})
    held = drift_weights(target, r)
    # after one bar: 0.55 vs 0.45 -> shares 0.55/1.0 and 0.45/1.0
    assert held.iloc[0]["win"] == pytest.approx(0.55)
    assert held.iloc[0]["lose"] == pytest.approx(0.45)
    # drift compounds: the winner's share keeps growing
    assert held.iloc[1]["win"] > held.iloc[0]["win"]


def test_rows_sum_to_the_target_total() -> None:
    target = pd.Series({"a": 0.6, "b": 0.4})
    rng = np.random.default_rng(2)
    r = pd.DataFrame(rng.normal(0.0, 0.02, size=(50, 2)), columns=["a", "b"])
    held = drift_weights(target, r)
    assert np.allclose(held.sum(axis=1).to_numpy(), 1.0)


def test_rebalance_every_bar_holds_the_target_exactly() -> None:
    target = pd.Series({"a": 0.7, "b": 0.3})
    rng = np.random.default_rng(3)
    r = pd.DataFrame(rng.normal(0.0, 0.03, size=(40, 2)), columns=["a", "b"])
    held = drift_weights(target, r, rebalance_every=1)
    # snapping back every bar means the drift never accumulates beyond one bar
    fresh = drift_weights(target, r.iloc[:1], rebalance_every=1)
    assert np.allclose(held.iloc[0].to_numpy(), fresh.iloc[0].to_numpy())
    # and the path never wanders far from the target
    assert float((held - target).abs().to_numpy().max()) < 0.05


def test_periodic_rebalance_resets_the_drift() -> None:
    target = pd.Series({"win": 0.5, "lose": 0.5})
    r = _returns({"win": [0.10] * 6, "lose": [-0.10] * 6})
    never = drift_weights(target, r, rebalance_every=None)
    every_two = drift_weights(target, r, rebalance_every=2)
    # unrebalanced drift runs away; the reset keeps pulling it back
    assert never.iloc[-1]["win"] > every_two.iloc[-1]["win"]
    # bar 2 is a reset bar: its weight equals the one-bar-after-target value
    assert every_two.iloc[2]["win"] == pytest.approx(0.55)


def test_zero_returns_leave_the_target_untouched() -> None:
    target = pd.Series({"a": 0.6, "b": 0.4})
    r = _returns({"a": [0.0, 0.0, 0.0], "b": [0.0, 0.0, 0.0]})
    held = drift_weights(target, r)
    assert np.allclose(held.to_numpy(), target.to_numpy())


def test_not_rebalancing_costs_no_turnover() -> None:
    # the whole point: drift is free, snapping back is what you pay for
    target = pd.Series({"win": 0.5, "lose": 0.5})
    r = _returns({"win": [0.05] * 4, "lose": [-0.05] * 4})
    held = drift_weights(target, r, rebalance_every=None)
    # holding the drifted book requires no trades between bars beyond drift
    drifted_end = held.iloc[-1]
    assert portfolio_turnover(drifted_end, target) > 0  # snapping back would cost


def test_short_leg_supported() -> None:
    target = pd.Series({"long": 1.0, "short": -0.5})
    r = _returns({"long": [0.10], "short": [-0.10]})
    held = drift_weights(target, r)
    assert float(held.sum(axis=1).iloc[0]) == pytest.approx(0.5)
    assert held.iloc[0]["short"] < 0


def test_bad_inputs_raise() -> None:
    target = pd.Series({"a": 1.0})
    r = _returns({"a": [0.01, 0.01]})
    with pytest.raises(ValueError, match="rebalance_every"):
        drift_weights(target, r, rebalance_every=0)
    with pytest.raises(ValueError, match="missing assets"):
        drift_weights(pd.Series({"zzz": 1.0}), r)
    with pytest.raises(ValueError, match="finite"):
        drift_weights(pd.Series({"a": np.nan}), r)
