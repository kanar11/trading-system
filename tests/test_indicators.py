"""Tests for the technical-indicators library."""

import numpy as np
import pandas as pd
import pytest

from src.indicators import (
    sma, ema, wma, vwma,
    rsi, macd, stochastic, williams_r, cci, roc,
    atr, bollinger, keltner, donchian,
    obv, vwap, chaikin_ad,
)


@pytest.fixture
def ohlcv():
    rng = np.random.default_rng(0)
    n = 200
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    closes = 100 + np.cumsum(rng.normal(0, 0.5, n))
    return pd.DataFrame(
        {
            "open": closes,
            "high": closes + np.abs(rng.normal(0, 0.5, n)),
            "low": closes - np.abs(rng.normal(0, 0.5, n)),
            "close": closes,
            "volume": rng.integers(1_000_000, 5_000_000, n),
        },
        index=dates,
    )


# ---------------------------------------------------------------------------
# trend
# ---------------------------------------------------------------------------

class TestTrend:
    def test_sma_equals_window_mean(self, ohlcv):
        s = sma(ohlcv["close"], 10)
        assert s.iloc[9] == pytest.approx(ohlcv["close"].iloc[:10].mean())

    def test_sma_nan_during_warmup(self, ohlcv):
        s = sma(ohlcv["close"], 10)
        assert s.iloc[:9].isna().all()

    def test_ema_response_lag(self, ohlcv):
        # ema reacts faster than sma — last-bar |ema - close| < |sma - close|
        last = -1
        s = sma(ohlcv["close"], 30).iloc[last]
        e = ema(ohlcv["close"], 30).iloc[last]
        target = ohlcv["close"].iloc[last]
        assert abs(e - target) <= abs(s - target) + 1e-9

    def test_wma_recent_weighted_more(self):
        # constant series → wma == constant; linear series → wma > sma
        s = pd.Series([float(i) for i in range(1, 21)])
        w = wma(s, 5).iloc[-1]
        m = sma(s, 5).iloc[-1]
        assert w > m  # rising series, weighted-recent gives larger value

    def test_vwma_equals_sma_when_volume_constant(self, ohlcv):
        v = pd.Series(1.0, index=ohlcv.index)
        w = vwma(ohlcv["close"], v, 20).iloc[-1]
        m = sma(ohlcv["close"], 20).iloc[-1]
        assert w == pytest.approx(m)

    def test_invalid_window_raises(self):
        with pytest.raises(ValueError):
            sma(pd.Series([1, 2, 3]), 0)
        with pytest.raises(ValueError):
            ema(pd.Series([1, 2, 3]), 0)


# ---------------------------------------------------------------------------
# momentum
# ---------------------------------------------------------------------------

class TestMomentum:
    def test_rsi_bounds(self, ohlcv):
        r = rsi(ohlcv["close"], 14).dropna()
        assert (r >= 0).all() and (r <= 100).all()

    def test_rsi_all_up_is_100(self):
        # strictly increasing series → all gains, no losses → RSI = 100
        s = pd.Series([float(i) for i in range(1, 50)])
        r = rsi(s, 14).iloc[-1]
        assert r == pytest.approx(100.0, abs=1e-6)

    def test_macd_columns(self, ohlcv):
        m = macd(ohlcv["close"])
        assert set(m.columns) == {"macd", "signal", "hist"}
        # hist == macd - signal by construction
        assert ((m["macd"] - m["signal"] - m["hist"]).abs().dropna() < 1e-12).all()

    def test_macd_fast_lt_slow_required(self, ohlcv):
        with pytest.raises(ValueError):
            macd(ohlcv["close"], fast=20, slow=10)

    def test_stochastic_bounds(self, ohlcv):
        st = stochastic(ohlcv["high"], ohlcv["low"], ohlcv["close"]).dropna()
        assert (st["k"] >= 0).all() and (st["k"] <= 100).all()

    def test_williams_r_range(self, ohlcv):
        w = williams_r(ohlcv["high"], ohlcv["low"], ohlcv["close"]).dropna()
        assert (w >= -100).all() and (w <= 0).all()

    def test_cci_finite(self, ohlcv):
        c = cci(ohlcv["high"], ohlcv["low"], ohlcv["close"]).dropna()
        assert np.isfinite(c).all()

    def test_roc_zero_for_flat_series(self):
        s = pd.Series([100.0] * 50)
        r = roc(s, 10).dropna()
        assert (r == 0).all()


# ---------------------------------------------------------------------------
# volatility
# ---------------------------------------------------------------------------

class TestVolatility:
    def test_atr_non_negative(self, ohlcv):
        a = atr(ohlcv["high"], ohlcv["low"], ohlcv["close"]).dropna()
        assert (a >= 0).all()

    def test_atr_smoothings_run(self, ohlcv):
        for smoothing in ("sma", "ema", "wilder"):
            v = atr(ohlcv["high"], ohlcv["low"], ohlcv["close"], smoothing=smoothing)
            assert v.notna().any()

    def test_atr_invalid_smoothing(self, ohlcv):
        with pytest.raises(ValueError):
            atr(ohlcv["high"], ohlcv["low"], ohlcv["close"], smoothing="bogus")

    def test_bollinger_band_ordering(self, ohlcv):
        b = bollinger(ohlcv["close"]).dropna()
        assert (b["upper"] >= b["middle"]).all()
        assert (b["middle"] >= b["lower"]).all()

    def test_bollinger_percent_b_range_typical(self, ohlcv):
        b = bollinger(ohlcv["close"]).dropna()
        # most observations land in [-0.5, 1.5]
        in_range = ((b["percent_b"] >= -0.5) & (b["percent_b"] <= 1.5)).sum()
        assert in_range / len(b) > 0.8

    def test_keltner_band_ordering(self, ohlcv):
        k = keltner(ohlcv["high"], ohlcv["low"], ohlcv["close"]).dropna()
        assert (k["upper"] >= k["middle"]).all()
        assert (k["middle"] >= k["lower"]).all()

    def test_donchian_upper_ge_lower(self, ohlcv):
        d = donchian(ohlcv["high"], ohlcv["low"], 20).dropna()
        assert (d["upper"] >= d["lower"]).all()


# ---------------------------------------------------------------------------
# volume
# ---------------------------------------------------------------------------

class TestVolume:
    def test_obv_starts_at_zero_then_moves(self, ohlcv):
        o = obv(ohlcv["close"], ohlcv["volume"])
        assert o.iloc[0] == 0
        assert o.diff().abs().sum() > 0

    def test_vwap_cumulative_when_anchor_none(self, ohlcv):
        v = vwap(ohlcv["close"], ohlcv["volume"], anchor=None)
        # the cumulative VWAP equals sum(pv) / sum(v) at every point
        manual = (ohlcv["close"] * ohlcv["volume"]).cumsum() / ohlcv["volume"].cumsum()
        pd.testing.assert_series_equal(
            v.reset_index(drop=True),
            manual.reset_index(drop=True),
            check_names=False,
        )

    def test_vwap_anchored_resets(self, ohlcv):
        # weekly anchor: first bar of each week starts a new VWAP
        v = vwap(ohlcv["close"], ohlcv["volume"], anchor="W")
        # first bar of each week's VWAP equals that bar's price
        first_per_week = v.groupby(ohlcv.index.to_period("W")).first()
        price_first_per_week = ohlcv["close"].groupby(ohlcv.index.to_period("W")).first()
        for w in first_per_week.index:
            assert first_per_week[w] == pytest.approx(price_first_per_week[w])

    def test_chaikin_ad_is_cumulative(self, ohlcv):
        ad = chaikin_ad(ohlcv["high"], ohlcv["low"], ohlcv["close"], ohlcv["volume"])
        # sum of diffs equals last - first (definition of cumulative)
        diffs = ad.diff().dropna()
        assert (ad.iloc[-1] - ad.iloc[0]) == pytest.approx(diffs.sum())
