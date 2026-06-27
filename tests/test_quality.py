"""Tests for the OHLCV data-quality auditor and cleaner."""

import numpy as np
import pandas as pd

from src.data.quality import check_ohlcv, clean_ohlcv


def _clean_frame(n: int = 60, seed: int = 0) -> pd.DataFrame:
    """A well-formed OHLCV frame: sorted, positive, OHLC-consistent, no stale runs."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    close = 100 + np.cumsum(rng.normal(0, 0.3, n))
    open_ = close + rng.normal(0, 0.1, n)
    high = np.maximum(open_, close) + np.abs(rng.normal(0, 0.2, n))
    low = np.minimum(open_, close) - np.abs(rng.normal(0, 0.2, n))
    volume = rng.integers(1_000_000, 5_000_000, n).astype(float)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=idx,
    )


def _col(df: pd.DataFrame, name: str) -> int:
    return int(df.columns.get_loc(name))


def test_clean_frame_reports_clean() -> None:
    rep = check_ohlcv(_clean_frame())
    assert rep.is_clean
    assert rep.issues == []
    assert rep.duplicate_timestamps == 0
    assert rep.unsorted_index is False
    assert rep.missing_values == 0
    assert rep.non_positive_prices == 0
    assert rep.ohlc_inconsistencies == 0
    assert rep.extreme_returns == 0
    assert rep.max_stale_run == 1


def test_detects_duplicate_timestamps() -> None:
    clean = _clean_frame(30)
    dup = pd.concat([clean, clean.iloc[[0]]]).sort_index()
    rep = check_ohlcv(dup)
    assert rep.duplicate_timestamps >= 1
    assert not rep.is_clean


def test_detects_unsorted_index() -> None:
    rep = check_ohlcv(_clean_frame(30).iloc[::-1])
    assert rep.unsorted_index is True
    assert not rep.is_clean


def test_detects_missing_values() -> None:
    df = _clean_frame(30).copy()
    df.iloc[3, _col(df, "close")] = np.nan
    assert check_ohlcv(df).missing_values >= 1


def test_detects_non_positive_prices() -> None:
    df = _clean_frame(30).copy()
    df.iloc[4, _col(df, "low")] = -5.0
    assert check_ohlcv(df).non_positive_prices >= 1


def test_detects_ohlc_inconsistency() -> None:
    df = _clean_frame(30).copy()
    df.iloc[7, _col(df, "high")] = df["low"].iloc[7] - 1.0  # high < low
    assert check_ohlcv(df).ohlc_inconsistencies >= 1


def test_detects_extreme_returns() -> None:
    df = _clean_frame(30).copy()
    df.iloc[10, _col(df, "close")] = df["close"].iloc[10] * 3  # +200% jump
    assert check_ohlcv(df).extreme_returns >= 1


def test_detects_stale_run() -> None:
    df = _clean_frame(40).copy()
    df.iloc[10:18, _col(df, "close")] = 123.0  # 8 identical closes
    rep = check_ohlcv(df, max_stale=5)
    assert rep.max_stale_run >= 8
    assert any("stale" in msg for msg in rep.issues)


def test_handles_partial_columns() -> None:
    idx = pd.date_range("2020-01-01", periods=20, freq="B")
    df = pd.DataFrame({"close": np.linspace(100, 110, 20)}, index=idx)
    rep = check_ohlcv(df)
    assert rep.ohlc_inconsistencies == 0
    assert isinstance(rep.is_clean, bool)


def test_clean_ohlcv_fixes_structural_issues() -> None:
    clean = _clean_frame(40)
    dirty = pd.concat([clean, clean.iloc[[0]]])  # duplicate timestamp
    dirty = dirty.iloc[::-1]  # unsort
    dirty.iloc[5, _col(dirty, "close")] = np.nan  # missing
    dirty.iloc[6, _col(dirty, "open")] = -1.0  # non-positive

    cleaned = clean_ohlcv(dirty)
    price_cols = ["open", "high", "low", "close"]
    assert cleaned.index.is_monotonic_increasing
    assert not cleaned.index.duplicated().any()
    assert cleaned[price_cols].notna().all().all()
    assert (cleaned[price_cols] > 0).all().all()


def test_clean_ohlcv_does_not_mutate_input() -> None:
    df = _clean_frame(20)
    snapshot = df.copy()
    clean_ohlcv(df)
    pd.testing.assert_frame_equal(df, snapshot)
