"""Tests for the data-layer extras: cache, resample, universe."""

import numpy as np
import pandas as pd
import pytest

from src.data.cache import CachedLoader, _make_key
from src.data.resample import resample_ohlcv, to_daily, to_weekly, to_monthly
from src.data.universe import (
    FAANG, DOW30, SECTOR_ETFS, BENCHMARKS, FACTOR_ETFS, get_universe,
)


# ---------------------------------------------------------------------------
# cache
# ---------------------------------------------------------------------------

class TestCache:
    def test_cache_hit_skips_loader(self, tmp_path):
        call_count = [0]

        def fake_loader(ticker, **kw):
            call_count[0] += 1
            dates = pd.date_range("2020-01-01", periods=5, freq="B")
            return pd.DataFrame(
                {"close": [100, 101, 102, 103, 104]}, index=dates,
            )

        loader = CachedLoader(fake_loader, cache_dir=tmp_path)
        loader("SPY", start="2020-01-01")
        loader("SPY", start="2020-01-01")
        assert call_count[0] == 1  # second call served from cache

    def test_cache_miss_for_different_kwargs(self, tmp_path):
        call_count = [0]

        def fake_loader(ticker, **kw):
            call_count[0] += 1
            return pd.DataFrame(
                {"close": [100, 101]},
                index=pd.date_range("2020-01-01", periods=2, freq="B"),
            )

        loader = CachedLoader(fake_loader, cache_dir=tmp_path)
        loader("SPY", start="2020-01-01")
        loader("SPY", start="2021-01-01")  # different start → cache miss
        assert call_count[0] == 2

    def test_cache_can_be_disabled(self, tmp_path):
        call_count = [0]

        def fake_loader(ticker, **kw):
            call_count[0] += 1
            return pd.DataFrame(
                {"close": [100]},
                index=pd.date_range("2020-01-01", periods=1, freq="B"),
            )

        loader = CachedLoader(fake_loader, cache_dir=tmp_path, enabled=False)
        loader("SPY")
        loader("SPY")
        assert call_count[0] == 2

    def test_cache_clear_removes_files(self, tmp_path):
        def fake_loader(ticker, **kw):
            return pd.DataFrame(
                {"close": [100]},
                index=pd.date_range("2020-01-01", periods=1, freq="B"),
            )

        loader = CachedLoader(fake_loader, cache_dir=tmp_path)
        loader("SPY", start="2020-01-01")
        loader("QQQ", start="2020-01-01")
        assert loader.clear() == 2  # both removed
        # next call is a fresh miss again
        loader("SPY", start="2020-01-01")
        assert any(p.exists() for p in tmp_path.iterdir())

    def test_make_key_deterministic(self):
        a = _make_key("SPY", start="2020-01-01", end="2024-01-01")
        b = _make_key("SPY", end="2024-01-01", start="2020-01-01")  # diff order
        assert a == b  # sorted kwargs

    def test_make_key_differs_by_kwargs(self):
        a = _make_key("SPY", start="2020-01-01")
        b = _make_key("SPY", start="2021-01-01")
        assert a != b


# ---------------------------------------------------------------------------
# resample
# ---------------------------------------------------------------------------

@pytest.fixture
def minute_ohlcv():
    """1-minute bars for ~6 trading hours."""
    rng = np.random.default_rng(0)
    n = 6 * 60  # 6 hours
    dates = pd.date_range("2020-01-02 09:30", periods=n, freq="1min")
    closes = 100 + np.cumsum(rng.normal(0, 0.05, n))
    return pd.DataFrame(
        {
            "open": closes,
            "high": closes + 0.1,
            "low": closes - 0.1,
            "close": closes,
            "volume": rng.integers(100, 10_000, n),
        },
        index=dates,
    )


class TestResample:
    def test_resample_5min_aggregation(self, minute_ohlcv):
        bars = resample_ohlcv(minute_ohlcv, "5min")
        assert len(bars) == len(minute_ohlcv) // 5
        # first 5-min bar's open == first minute's open
        assert bars["open"].iloc[0] == minute_ohlcv["open"].iloc[0]
        assert bars["close"].iloc[0] == minute_ohlcv["close"].iloc[4]
        # high is max of source bars
        assert bars["high"].iloc[0] == minute_ohlcv["high"].iloc[:5].max()
        # volume sums
        assert bars["volume"].iloc[0] == minute_ohlcv["volume"].iloc[:5].sum()

    def test_to_daily(self, minute_ohlcv):
        d = to_daily(minute_ohlcv)
        # 6 hours fits inside one trading day
        assert len(d) == 1
        assert d["volume"].iloc[0] == minute_ohlcv["volume"].sum()

    def test_resample_requires_datetime_index(self):
        df = pd.DataFrame({"close": [1, 2, 3]})
        with pytest.raises(ValueError, match="DatetimeIndex"):
            resample_ohlcv(df, "1D")

    def test_resample_requires_ohlcv_column(self):
        dates = pd.date_range("2020-01-01", periods=5, freq="B")
        df = pd.DataFrame({"other": [1, 2, 3, 4, 5]}, index=dates)
        with pytest.raises(ValueError, match="open/high/low/close/volume"):
            resample_ohlcv(df, "1W")

    def test_weekly_monthly_convenience(self, minute_ohlcv):
        # build a multi-week frame
        df = pd.concat([minute_ohlcv.shift(freq=f"{d}D") for d in range(0, 30, 1)])
        df = df[~df.index.duplicated(keep="first")].sort_index()
        w = to_weekly(df)
        m = to_monthly(df)
        assert len(w) >= 1
        assert len(m) >= 1


# ---------------------------------------------------------------------------
# universe
# ---------------------------------------------------------------------------

class TestUniverse:
    def test_constants_non_empty(self):
        assert len(FAANG) == 5
        assert len(DOW30) == 30
        assert len(SECTOR_ETFS) == 11
        assert len(BENCHMARKS) > 0
        assert len(FACTOR_ETFS) > 0

    def test_get_universe_by_name(self):
        assert set(get_universe("faang")) == set(FAANG)
        assert set(get_universe("dow30")) == set(DOW30)
        assert set(get_universe("SECTORS")) == set(SECTOR_ETFS.keys())  # case-insensitive

    def test_get_universe_unknown_raises(self):
        with pytest.raises(KeyError, match="Unknown"):
            get_universe("nonexistent")

    def test_get_universe_returns_copy(self):
        a = get_universe("faang")
        a.append("EXTRA")
        b = get_universe("faang")
        assert "EXTRA" not in b  # original list not mutated
