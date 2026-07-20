"""Tests for floor-trader pivot points."""

import numpy as np
import pandas as pd
import pytest

from src.indicators import pivot_points


def _bars(highs: list[float], lows: list[float], closes: list[float]) -> tuple[pd.Series, ...]:
    idx = pd.date_range("2024-01-01", periods=len(highs), freq="B")
    return (
        pd.Series(highs, index=idx),
        pd.Series(lows, index=idx),
        pd.Series(closes, index=idx),
    )


def test_classic_levels_by_hand() -> None:
    # previous bar: H=110, L=90, C=105 -> P = 305/3
    high, low, close = _bars([110.0, 108.0], [90.0, 95.0], [105.0, 100.0])
    out = pivot_points(high, low, close, method="classic")
    p = (110.0 + 90.0 + 105.0) / 3.0
    assert out["pivot"].iloc[1] == pytest.approx(p)
    assert out["r1"].iloc[1] == pytest.approx(2 * p - 90.0)
    assert out["s1"].iloc[1] == pytest.approx(2 * p - 110.0)
    assert out["r2"].iloc[1] == pytest.approx(p + 20.0)
    assert out["s2"].iloc[1] == pytest.approx(p - 20.0)


def test_columns_and_ordering() -> None:
    rng = np.random.default_rng(4)
    n = 60
    close = pd.Series(100.0 + rng.normal(0, 1, n).cumsum())
    high = close + np.abs(rng.normal(1, 0.3, n))
    low = close - np.abs(rng.normal(1, 0.3, n))
    out = pivot_points(high, low, close)
    assert list(out.columns) == ["pivot", "r1", "r2", "r3", "s1", "s2", "s3"]
    valid = out.dropna()
    # resistances above the pivot, supports below, monotone outward
    assert (valid["r1"] >= valid["pivot"]).all()
    assert (valid["r2"] >= valid["r1"]).all()
    assert (valid["s1"] <= valid["pivot"]).all()
    assert (valid["s2"] <= valid["s1"]).all()


def test_is_causal_first_bar_is_nan() -> None:
    high, low, close = _bars([110.0, 108.0, 106.0], [90.0, 95.0, 94.0], [105.0, 100.0, 99.0])
    out = pivot_points(high, low, close)
    assert out.iloc[0].isna().all()  # no prior bar to derive levels from
    # bar 1's levels use bar 0's HLC only, never bar 1's own values
    p = (110.0 + 90.0 + 105.0) / 3.0
    assert out["pivot"].iloc[1] == pytest.approx(p)


def test_woodie_weights_the_close() -> None:
    high, low, close = _bars([110.0, 108.0], [90.0, 95.0], [108.0, 100.0])
    classic = pivot_points(high, low, close, method="classic")
    woodie = pivot_points(high, low, close, method="woodie")
    # close (108) is above the classic pivot, so weighting it lifts the pivot
    assert woodie["pivot"].iloc[1] == pytest.approx((110.0 + 90.0 + 2 * 108.0) / 4.0)
    assert woodie["pivot"].iloc[1] > classic["pivot"].iloc[1]


def test_fibonacci_uses_retracement_bands() -> None:
    high, low, close = _bars([110.0, 108.0], [90.0, 95.0], [105.0, 100.0])
    out = pivot_points(high, low, close, method="fibonacci")
    p = (110.0 + 90.0 + 105.0) / 3.0
    rng = 20.0
    assert out["r1"].iloc[1] == pytest.approx(p + 0.382 * rng)
    assert out["r2"].iloc[1] == pytest.approx(p + 0.618 * rng)
    assert out["r3"].iloc[1] == pytest.approx(p + rng)


def test_unknown_method_raises() -> None:
    high, low, close = _bars([110.0, 108.0], [90.0, 95.0], [105.0, 100.0])
    with pytest.raises(ValueError, match="method"):
        pivot_points(high, low, close, method="camarilla")
