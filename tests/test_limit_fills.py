"""Tests for vectorised limit-order fill simulation."""

import numpy as np
import pandas as pd
import pytest

from src.execution import simulate_limit_fills


def _bars() -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=4, freq="B")
    #                bar:      0       1       2       3
    return pd.DataFrame(
        {
            "open": [100.0, 101.0, 95.0, 103.0],
            "high": [102.0, 104.0, 99.0, 106.0],
            "low": [99.0, 100.5, 94.0, 102.0],
        },
        index=idx,
    )


def test_buy_limit_fills_when_low_touches() -> None:
    out = simulate_limit_fills(_bars(), 99.5, side="buy")
    assert list(out.columns) == ["filled", "fill_price"]
    assert list(out["filled"]) == [True, False, True, False]


def test_buy_gap_through_open_fills_at_open() -> None:
    out = simulate_limit_fills(_bars(), 99.5, side="buy")
    # bar 2 opens at 95, below the 99.5 limit -> price improvement at open
    assert out["fill_price"].iloc[2] == pytest.approx(95.0)
    # bar 0 trades down to the limit intrabar -> fill at the limit itself
    assert out["fill_price"].iloc[0] == pytest.approx(99.5)


def test_sell_limit_symmetric() -> None:
    out = simulate_limit_fills(_bars(), 103.5, side="sell")
    assert list(out["filled"]) == [False, True, False, True]
    assert out["fill_price"].iloc[1] == pytest.approx(103.5)  # touched intrabar
    # bar 3 opens at 103.0 below the limit; high 106 touches -> limit price
    assert out["fill_price"].iloc[3] == pytest.approx(103.5)


def test_sell_open_above_limit_fills_at_open() -> None:
    out = simulate_limit_fills(_bars(), 102.5, side="sell")
    assert bool(out["filled"].iloc[3])
    assert out["fill_price"].iloc[3] == pytest.approx(103.0)  # opens through


def test_unfilled_bars_have_nan_price() -> None:
    out = simulate_limit_fills(_bars(), 90.0, side="buy")
    assert not out["filled"].any()
    assert out["fill_price"].isna().all()


def test_per_bar_limit_series_and_nan_means_no_order() -> None:
    df = _bars()
    limits = pd.Series([99.5, np.nan, 94.5, np.nan], index=df.index)
    out = simulate_limit_fills(df, limits, side="buy")
    assert list(out["filled"]) == [True, False, True, False]
    assert out["fill_price"].iloc[2] == pytest.approx(94.5)  # open 95 > limit


def test_buy_fills_never_worse_than_limit() -> None:
    rng = np.random.default_rng(9)
    n = 300
    close = 100.0 * np.cumprod(1 + rng.normal(0, 0.01, n))
    idx = pd.date_range("2023-01-02", periods=n, freq="B")
    df = pd.DataFrame(
        {
            "open": close * (1 + rng.normal(0, 0.002, n)),
            "high": close * (1 + np.abs(rng.normal(0, 0.006, n))),
            "low": close * (1 - np.abs(rng.normal(0, 0.006, n))),
        },
        index=idx,
    )
    limits = pd.Series(df["low"].shift(1), index=idx)  # buy yesterday's low
    out = simulate_limit_fills(df, limits, side="buy")
    filled = out["filled"].to_numpy()
    assert filled.any()
    assert (out["fill_price"].to_numpy()[filled] <= limits.to_numpy()[filled] + 1e-12).all()


def test_bad_inputs_raise() -> None:
    df = _bars()
    with pytest.raises(ValueError, match="side"):
        simulate_limit_fills(df, 100.0, side="hold")
    with pytest.raises(ValueError, match="columns"):
        simulate_limit_fills(df.drop(columns=["low"]), 100.0)
    with pytest.raises(ValueError, match="> 0"):
        simulate_limit_fills(df, -5.0)
    misaligned = pd.Series([100.0], index=df.index[:1])
    with pytest.raises(ValueError, match="index"):
        simulate_limit_fills(df, misaligned)
    bad = df.copy()
    bad.iloc[1, 0] = np.nan
    with pytest.raises(ValueError, match="positive"):
        simulate_limit_fills(bad, 100.0)
