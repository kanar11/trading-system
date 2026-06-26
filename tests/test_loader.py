"""Tests for the Yahoo Finance data loader."""

from collections.abc import Callable

import numpy as np
import pandas as pd
import pytest

from quantbt.data import loader
from quantbt.data.loader import load_yahoo_ohlcv

SetFrame = Callable[[pd.DataFrame], None]


def _fake_yahoo_frame(n: int = 30, multiindex: bool = False) -> pd.DataFrame:
    """Build a yfinance-style frame with Title-case columns."""
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    close = np.linspace(100, 110, n)
    data = {
        "Open": close * 0.99,
        "High": close * 1.01,
        "Low": close * 0.98,
        "Close": close,
        "Volume": np.full(n, 1_000_000),
    }
    df = pd.DataFrame(data, index=dates)
    if multiindex:
        df.columns = pd.MultiIndex.from_product([df.columns, ["SPY"]])
    return df


@pytest.fixture
def patched_download(monkeypatch: pytest.MonkeyPatch) -> SetFrame:
    """Patch yf.download with a configurable fake; returns a setter."""

    def _set(frame: pd.DataFrame) -> None:
        monkeypatch.setattr(loader.yf, "download", lambda *a, **k: frame)

    return _set


def test_returns_lowercase_ohlcv(patched_download: SetFrame) -> None:
    patched_download(_fake_yahoo_frame())
    df = load_yahoo_ohlcv("spy")
    assert list(df.columns) == ["open", "high", "low", "close", "volume"]
    assert isinstance(df.index, pd.DatetimeIndex)
    assert df.index.is_monotonic_increasing


def test_flattens_multiindex_columns(patched_download: SetFrame) -> None:
    patched_download(_fake_yahoo_frame(multiindex=True))
    df = load_yahoo_ohlcv("SPY")
    assert list(df.columns) == ["open", "high", "low", "close", "volume"]


def test_empty_download_raises(patched_download: SetFrame) -> None:
    patched_download(pd.DataFrame())
    with pytest.raises(ValueError, match="No data returned"):
        load_yahoo_ohlcv("SPY")


def test_missing_columns_raises(patched_download: SetFrame) -> None:
    frame = _fake_yahoo_frame().drop(columns=["Volume"])
    patched_download(frame)
    with pytest.raises(ValueError, match="Missing columns"):
        load_yahoo_ohlcv("SPY")


def test_empty_ticker_raises() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        load_yahoo_ohlcv("   ")


def test_invalid_start_date_raises() -> None:
    with pytest.raises(ValueError, match="Invalid start date"):
        load_yahoo_ohlcv("SPY", start="not-a-date")


def test_end_before_start_raises() -> None:
    with pytest.raises(ValueError, match="must be after start"):
        load_yahoo_ohlcv("SPY", start="2020-01-01", end="2019-01-01")


def test_download_failure_wrapped(monkeypatch: pytest.MonkeyPatch) -> None:
    def _boom(*args: object, **kwargs: object) -> pd.DataFrame:
        raise ConnectionError("network down")

    monkeypatch.setattr(loader.yf, "download", _boom)
    with pytest.raises(RuntimeError, match="Failed to download"):
        load_yahoo_ohlcv("SPY")


def test_drops_incomplete_rows(patched_download: SetFrame) -> None:
    frame = _fake_yahoo_frame()
    frame.loc[frame.index[0], "Close"] = np.nan
    patched_download(frame)
    df = load_yahoo_ohlcv("SPY")
    assert not df.isna().any().any()
