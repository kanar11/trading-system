"""Tests for the wide price-panel builder."""

import numpy as np
import pandas as pd
import pytest

from src.backtest import backtest_weights
from src.data.panel import build_close_frame


def _ohlcv(idx: pd.DatetimeIndex, base: float) -> pd.DataFrame:
    close = base + np.arange(len(idx), dtype=float)
    return pd.DataFrame(
        {"open": close - 0.5, "high": close + 1, "low": close - 1, "close": close},
        index=idx,
    )


def test_common_calendar_gives_identical_inner_and_outer() -> None:
    idx = pd.date_range("2024-01-01", periods=10, freq="B")
    frames = {"aaa": _ohlcv(idx, 100.0), "bbb": _ohlcv(idx, 50.0)}
    inner = build_close_frame(frames, join="inner")
    outer = build_close_frame(frames, join="outer")
    assert list(inner.columns) == ["aaa", "bbb"]
    assert inner.equals(outer)
    assert inner.loc[idx[3], "bbb"] == pytest.approx(53.0)


def test_inner_join_drops_missing_sessions() -> None:
    idx = pd.date_range("2024-01-01", periods=10, freq="B")
    holey = _ohlcv(idx, 50.0).drop(index=idx[4])  # bbb missed one session
    frames = {"aaa": _ohlcv(idx, 100.0), "bbb": holey}
    inner = build_close_frame(frames, join="inner")
    assert len(inner) == 9
    assert idx[4] not in inner.index
    assert not inner.isna().any().any()


def test_outer_join_keeps_gap_and_ffill_fills_it() -> None:
    idx = pd.date_range("2024-01-01", periods=10, freq="B")
    holey = _ohlcv(idx, 50.0).drop(index=idx[4])
    frames = {"aaa": _ohlcv(idx, 100.0), "bbb": holey}
    outer = build_close_frame(frames, join="outer")
    assert np.isnan(outer.loc[idx[4], "bbb"])
    filled = build_close_frame(frames, join="outer", ffill_limit=1)
    assert filled.loc[idx[4], "bbb"] == pytest.approx(53.0)  # previous close


def test_leading_nans_are_never_filled() -> None:
    idx = pd.date_range("2024-01-01", periods=10, freq="B")
    late = _ohlcv(idx[5:], 50.0)  # bbb starts listing mid-sample
    frames = {"aaa": _ohlcv(idx, 100.0), "bbb": late}
    filled = build_close_frame(frames, join="outer", ffill_limit=5)
    assert filled["bbb"].iloc[:5].isna().all()


def test_column_argument_extracts_other_fields() -> None:
    idx = pd.date_range("2024-01-01", periods=5, freq="B")
    frames = {"aaa": _ohlcv(idx, 100.0)}
    highs = build_close_frame(frames, column="high")
    assert highs.loc[idx[0], "aaa"] == pytest.approx(101.0)


def test_panel_feeds_the_weights_engine() -> None:
    idx = pd.date_range("2024-01-01", periods=40, freq="B")
    frames = {"aaa": _ohlcv(idx, 100.0), "bbb": _ohlcv(idx, 50.0)}
    prices = build_close_frame(frames)
    weights = pd.DataFrame(0.5, index=prices.index, columns=prices.columns)
    result = backtest_weights(prices, weights, cost_bps=0.0)
    assert np.isfinite(result["equity_curve"].to_numpy()).all()


def test_bad_inputs_raise() -> None:
    idx = pd.date_range("2024-01-01", periods=5, freq="B")
    good = _ohlcv(idx, 100.0)
    with pytest.raises(ValueError, match="empty"):
        build_close_frame({})
    with pytest.raises(ValueError, match="join"):
        build_close_frame({"aaa": good}, join="left")
    with pytest.raises(ValueError, match="ffill_limit"):
        build_close_frame({"aaa": good}, ffill_limit=0)
    with pytest.raises(ValueError, match="missing column"):
        build_close_frame({"aaa": good.drop(columns=["close"])})
    with pytest.raises(TypeError, match="DatetimeIndex"):
        build_close_frame({"aaa": good.reset_index(drop=True)})
    unsorted = good.iloc[::-1]
    with pytest.raises(ValueError, match="unsorted"):
        build_close_frame({"aaa": unsorted})
    other = _ohlcv(pd.date_range("2030-01-01", periods=5, freq="B"), 50.0)
    with pytest.raises(ValueError, match="no common timestamps"):
        build_close_frame({"aaa": good, "bbb": other}, join="inner")
