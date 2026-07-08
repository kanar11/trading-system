"""Tests for corporate-action back-adjustment."""

import numpy as np
import pandas as pd
import pytest

from src.data.corporate_actions import adjust_ohlcv, adjustment_factors


def _close(values: list[float], start: str = "2024-01-01") -> pd.Series:
    idx = pd.date_range(start, periods=len(values), freq="B")
    return pd.Series(values, index=idx, name="close")


def test_two_for_one_split_halves_prior_bars() -> None:
    close = _close([100.0, 102.0, 51.0, 52.0])
    split_date = close.index[2]
    factors = adjustment_factors(close, splits={split_date: 2.0})
    adjusted = close * factors
    assert np.allclose(adjusted.to_numpy(), [50.0, 51.0, 51.0, 52.0])
    assert factors.iloc[-1] == 1.0
    assert factors.name == "adj_factor"


def test_dividend_makes_ex_date_return_continuous() -> None:
    # price drops exactly by the dividend -> adjusted return across ex-date is 0
    close = _close([100.0, 99.0, 99.0])
    ex_date = close.index[1]
    factors = adjustment_factors(close, dividends={ex_date: 1.0})
    adjusted = close * factors
    assert adjusted.iloc[0] == pytest.approx(99.0)  # 100 * (1 - 1/100)
    assert adjusted.pct_change().iloc[1] == pytest.approx(0.0)


def test_split_and_dividend_compose_multiplicatively() -> None:
    close = _close([100.0, 100.0, 50.0, 50.0])
    split_date = close.index[2]
    ex_date = close.index[1]
    factors = adjustment_factors(close, splits={split_date: 2.0}, dividends={ex_date: 2.0})
    # bar 0: split factor 1/2 and dividend factor (1 - 2/100) = 0.98
    assert factors.iloc[0] == pytest.approx(0.5 * 0.98)
    # bar 1: after the ex-date, before the split -> split factor only
    assert factors.iloc[1] == pytest.approx(0.5)
    assert factors.iloc[2] == 1.0


def test_returns_after_last_event_are_unchanged() -> None:
    close = _close([100.0, 102.0, 51.0, 52.5, 53.0])
    adjusted = close * adjustment_factors(close, splits={close.index[2]: 2.0})
    raw_tail = close.pct_change().iloc[3:]
    adj_tail = adjusted.pct_change().iloc[3:]
    assert np.allclose(raw_tail.to_numpy(), adj_tail.to_numpy())


def test_event_before_first_bar_is_a_no_op() -> None:
    close = _close([100.0, 101.0])
    early = pd.Timestamp("2023-01-01")
    factors = adjustment_factors(close, splits={early: 2.0}, dividends={early: 1.0})
    assert (factors == 1.0).all()


def test_adjust_ohlcv_scales_prices_and_volume() -> None:
    idx = pd.date_range("2024-01-01", periods=4, freq="B")
    df = pd.DataFrame(
        {
            "open": [100.0, 101.0, 50.5, 51.0],
            "high": [101.0, 103.0, 51.5, 52.0],
            "low": [99.0, 100.0, 50.0, 50.5],
            "close": [100.0, 102.0, 51.0, 51.5],
            "volume": [1_000.0, 1_100.0, 2_400.0, 2_500.0],
        },
        index=idx,
    )
    out = adjust_ohlcv(df, splits={idx[2]: 2.0})
    assert out["close"].iloc[0] == pytest.approx(50.0)
    assert out["open"].iloc[0] == pytest.approx(50.0)
    # OHLC ordering survives the uniform scaling
    assert (out["high"] >= out["low"]).all()
    # pre-split volume is doubled, post-split untouched
    assert out["volume"].iloc[0] == pytest.approx(2_000.0)
    assert out["volume"].iloc[3] == pytest.approx(2_500.0)
    # input not mutated
    assert df["close"].iloc[0] == 100.0


def test_dividend_does_not_touch_volume() -> None:
    idx = pd.date_range("2024-01-01", periods=3, freq="B")
    df = pd.DataFrame({"close": [100.0, 99.0, 99.5], "volume": [1_000.0] * 3}, index=idx)
    out = adjust_ohlcv(df, dividends={idx[1]: 1.0})
    assert np.allclose(out["volume"].to_numpy(), 1_000.0)
    assert out["close"].iloc[0] == pytest.approx(99.0)


def test_missing_close_raises() -> None:
    idx = pd.date_range("2024-01-01", periods=2, freq="B")
    with pytest.raises(ValueError, match="close"):
        adjust_ohlcv(pd.DataFrame({"open": [1.0, 2.0]}, index=idx))


def test_invalid_events_raise() -> None:
    close = _close([100.0, 101.0, 102.0])
    with pytest.raises(ValueError, match="ratio"):
        adjustment_factors(close, splits={close.index[1]: 0.0})
    with pytest.raises(ValueError, match="dividend"):
        adjustment_factors(close, dividends={close.index[1]: -1.0})
    with pytest.raises(ValueError, match="preceding close"):
        adjustment_factors(close, dividends={close.index[1]: 100.0})


def test_invalid_series_raises() -> None:
    with pytest.raises(TypeError, match="DatetimeIndex"):
        adjustment_factors(pd.Series([1.0, 2.0]))
    unsorted = pd.Series([1.0, 2.0], index=pd.DatetimeIndex(["2024-01-02", "2024-01-01"]))
    with pytest.raises(ValueError, match="sorted"):
        adjustment_factors(unsorted)
    bad = _close([100.0, -1.0])
    with pytest.raises(ValueError, match="positive"):
        adjustment_factors(bad)
