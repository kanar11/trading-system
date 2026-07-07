"""Tests for short-borrow and margin financing costs."""

import numpy as np
import pandas as pd
import pytest

from src.execution import apply_financing, financing_costs


def _idx(n: int) -> pd.DatetimeIndex:
    return pd.date_range("2024-01-01", periods=n, freq="B")


def test_unlevered_long_only_is_free() -> None:
    positions = pd.Series([0.0, 0.5, 1.0, 1.0], index=_idx(4))
    costs = financing_costs(positions, borrow_rate=0.03, margin_rate=0.05)
    assert costs.name == "financing_cost"
    assert (costs == 0.0).all()


def test_full_short_pays_borrow_rate_per_bar() -> None:
    positions = pd.Series([-1.0] * 5, index=_idx(5))
    costs = financing_costs(positions, borrow_rate=0.0252, periods_per_year=252)
    assert np.allclose(costs.to_numpy(), 0.0001)  # 2.52% / 252 per bar


def test_two_x_long_pays_margin_on_the_excess() -> None:
    positions = pd.Series([2.0] * 4, index=_idx(4))
    costs = financing_costs(positions, margin_rate=0.0504, periods_per_year=252)
    # financed notional = 1.0 -> 5.04%/252 = 2 bps per bar
    assert np.allclose(costs.to_numpy(), 0.0002)


def test_short_leg_pays_both_borrow_and_margin() -> None:
    # 100/100 long-short: gross = 2 -> 1.0 financed; short notional = 1.0
    positions = pd.DataFrame({"a": [1.0], "b": [-1.0]}, index=_idx(1))
    costs = financing_costs(positions, borrow_rate=0.0252, margin_rate=0.0252)
    assert np.allclose(costs.to_numpy(), 0.0001 + 0.0001)


def test_frame_input_sums_across_assets() -> None:
    positions = pd.DataFrame({"a": [-0.3, 0.0], "b": [-0.2, 0.0]}, index=_idx(2))
    costs = financing_costs(positions, borrow_rate=0.252, periods_per_year=252)
    # bar 0: short notional 0.5 -> 0.5 * 0.001; bar 1: flat
    assert costs.iloc[0] == pytest.approx(0.0005)
    assert costs.iloc[1] == 0.0


def test_apply_financing_subtracts_the_drag() -> None:
    idx = _idx(3)
    returns = pd.Series([0.01, 0.02, -0.01], index=idx)
    positions = pd.Series([-1.0, -1.0, -1.0], index=idx)
    net = apply_financing(returns, positions, borrow_rate=0.0252)
    assert np.allclose(net.to_numpy(), returns.to_numpy() - 0.0001)


def test_zero_rates_cost_nothing() -> None:
    positions = pd.Series([-2.0, 3.0], index=_idx(2))
    assert (financing_costs(positions) == 0.0).all()


def test_negative_rate_raises() -> None:
    positions = pd.Series([1.0], index=_idx(1))
    with pytest.raises(ValueError, match="borrow_rate"):
        financing_costs(positions, borrow_rate=-0.01)
    with pytest.raises(ValueError, match="margin_rate"):
        financing_costs(positions, margin_rate=-0.01)
    with pytest.raises(ValueError, match="periods_per_year"):
        financing_costs(positions, periods_per_year=0)


def test_nan_positions_raise() -> None:
    positions = pd.Series([1.0, np.nan], index=_idx(2))
    with pytest.raises(ValueError, match="NaN"):
        financing_costs(positions, borrow_rate=0.01)


def test_index_mismatch_raises() -> None:
    returns = pd.Series([0.01], index=_idx(1))
    positions = pd.Series([1.0, 1.0], index=_idx(2))
    with pytest.raises(ValueError, match="index"):
        apply_financing(returns, positions)
