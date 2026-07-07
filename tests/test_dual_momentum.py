"""Tests for the dual momentum rotation strategy."""

import numpy as np
import pandas as pd
import pytest

from src.data.calendar import rebalance_mask
from src.strategy.dual_momentum import dual_momentum_strategy


def _prices(n: int = 130) -> pd.DataFrame:
    """Two clean trends: 'up' compounds +0.4%/bar, 'down' -0.4%/bar."""
    idx = pd.date_range("2024-01-01", periods=n, freq="B")
    up = 100.0 * np.cumprod(np.full(n, 1.004))
    down = 100.0 * np.cumprod(np.full(n, 0.996))
    return pd.DataFrame({"up": up, "down": down}, index=idx)


def test_picks_the_relative_winner() -> None:
    weights = dual_momentum_strategy(_prices(), lookback=20, top_n=1)
    # after warm-up and the first rebalance, all weight sits on the riser
    tail = weights.iloc[-20:]
    assert (tail["up"] == 1.0).all()
    assert (tail["down"] == 0.0).all()


def test_absolute_veto_goes_to_cash() -> None:
    df = _prices()
    both_down = pd.DataFrame({"a": df["down"], "b": df["down"] * 0.98}, index=df.index)
    weights = dual_momentum_strategy(both_down, lookback=20, top_n=1)
    assert (weights.to_numpy() == 0.0).all()


def test_rows_sum_to_at_most_one_and_non_negative() -> None:
    weights = dual_momentum_strategy(_prices(), lookback=20, top_n=2)
    arr = weights.to_numpy()
    assert (arr >= 0.0).all()
    assert (arr.sum(axis=1) <= 1.0 + 1e-12).all()


def test_top_two_split_equally() -> None:
    df = _prices()
    df["up2"] = df["up"] * 0.999  # a second, slightly weaker riser
    weights = dual_momentum_strategy(df, lookback=20, top_n=2)
    tail = weights.iloc[-10:]
    assert (tail["up"] == 0.5).all()
    assert (tail["up2"] == 0.5).all()
    assert (tail["down"] == 0.0).all()


def test_partial_fill_leaves_slot_in_cash() -> None:
    # top_n=2 but only one asset passes the absolute filter -> row sums 0.5
    weights = dual_momentum_strategy(_prices(), lookback=20, top_n=2)
    tail = weights.iloc[-10:]
    assert (tail["up"] == 0.5).all()
    assert (tail["down"] == 0.0).all()


def test_all_cash_before_first_valid_rebalance() -> None:
    df = _prices()
    weights = dual_momentum_strategy(df, lookback=40, top_n=1)
    # January's month-end falls inside the 40-bar warm-up: no momentum
    # reading yet, so the strategy stays in cash
    jan = weights[weights.index < "2024-02-01"]
    assert (jan.to_numpy() == 0.0).all()


def test_weights_change_only_on_rebalance_dates() -> None:
    df = _prices()
    weights = dual_momentum_strategy(df, lookback=20, top_n=1)
    changed = (weights.diff().abs().sum(axis=1) > 0).to_numpy()
    month_ends = rebalance_mask(pd.DatetimeIndex(df.index), freq="M").to_numpy()
    assert not changed[~month_ends].any()


def test_weekly_rebalance_reacts_faster_than_monthly() -> None:
    # lookback=25 ends the warm-up just after January's month-end (bar 22),
    # so the monthly variant must wait for Feb 29 while the weekly one can
    # already invest on the first Friday after the momentum series starts
    df = _prices()
    weekly = dual_momentum_strategy(df, lookback=25, top_n=1, rebalance="W")
    monthly = dual_momentum_strategy(df, lookback=25, top_n=1, rebalance="M")
    first_weekly = (weekly["up"] > 0).idxmax()
    first_monthly = (monthly["up"] > 0).idxmax()
    assert first_weekly < first_monthly


def test_bad_params_raise() -> None:
    df = _prices()
    with pytest.raises(ValueError, match="lookback"):
        dual_momentum_strategy(df, lookback=0)
    with pytest.raises(ValueError, match="top_n"):
        dual_momentum_strategy(df, top_n=0)
    with pytest.raises(ValueError, match="column"):
        dual_momentum_strategy(df[[]])
