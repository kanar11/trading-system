"""Tests for KAMA and Parabolic SAR."""

import numpy as np
import pandas as pd
import pytest

from src.indicators import kama, parabolic_sar


def _line(n: int = 60, start: float = 100.0, step: float = 1.0) -> pd.Series:
    return pd.Series(start + step * np.arange(n, dtype=float))


# --- KAMA -------------------------------------------------------------------


def test_kama_warm_up_and_seed() -> None:
    series = _line()
    out = kama(series, er_period=10)
    assert out.iloc[:10].isna().all()
    assert out.iloc[10] == series.iloc[10]  # seeded with the price


def test_kama_constant_series_stays_put() -> None:
    series = pd.Series(np.full(40, 50.0))
    out = kama(series, er_period=10)
    assert (out.iloc[10:] == 50.0).all()


def test_kama_follows_a_clean_trend_quickly() -> None:
    series = _line(80)
    out = kama(series, er_period=10, fast=2, slow=30)
    # on a perfect trend ER = 1, so KAMA uses the fast EMA(2) constant and
    # converges to a small constant lag behind the line
    tail_lag = (series - out).iloc[-20:]
    assert (tail_lag > 0).all()
    assert (tail_lag < 2.0).all()
    assert out.iloc[11:].is_monotonic_increasing


def test_kama_slower_in_chop_than_in_trend() -> None:
    rng = np.random.default_rng(4)
    chop = pd.Series(100.0 + rng.normal(0, 1.0, 200)).round(2)
    trend = _line(200, step=0.5)
    lag_chop = kama(chop, er_period=10)
    lag_trend = kama(trend, er_period=10)
    # in chop the adaptive constant collapses: per-bar KAMA moves are much
    # smaller than the price moves themselves
    move_ratio = float(lag_chop.diff().abs().iloc[20:].mean() / chop.diff().abs().iloc[20:].mean())
    trend_ratio = float(
        lag_trend.diff().abs().iloc[20:].mean() / trend.diff().abs().iloc[20:].mean()
    )
    assert move_ratio < 0.35
    assert trend_ratio > 0.9


def test_kama_bad_params_raise() -> None:
    series = _line()
    with pytest.raises(ValueError, match="er_period"):
        kama(series, er_period=0)
    with pytest.raises(ValueError, match="fast"):
        kama(series, fast=30, slow=30)


# --- Parabolic SAR ----------------------------------------------------------


def _trend_bars(n: int = 50, step: float = 1.0) -> tuple[pd.Series, pd.Series]:
    base = 100.0 + step * np.arange(n, dtype=float)
    return pd.Series(base + 0.5), pd.Series(base - 0.5)


def test_psar_stays_below_price_in_uptrend() -> None:
    high, low = _trend_bars(step=1.0)
    out = parabolic_sar(high, low)
    assert list(out.columns) == ["sar", "trend"]
    assert np.isnan(out["sar"].iloc[0])
    assert (out["trend"].iloc[1:] == 1).all()
    assert (out["sar"].iloc[1:] < low.iloc[1:]).all()
    # the stop only ratchets up in an up-trend
    assert out["sar"].iloc[2:].is_monotonic_increasing


def test_psar_stays_above_price_in_downtrend() -> None:
    high, low = _trend_bars(step=-1.0)
    out = parabolic_sar(high, low)
    assert (out["trend"].iloc[1:] == -1).all()
    assert (out["sar"].iloc[1:] > high.iloc[1:]).all()


def test_psar_reverses_on_a_v_shape() -> None:
    down = 100.0 - np.arange(25, dtype=float)
    upswing = down[-1] + np.arange(1, 26, dtype=float)
    base = np.concatenate([down, upswing])
    out = parabolic_sar(pd.Series(base + 0.5), pd.Series(base - 0.5))
    trend = out["trend"].to_numpy()
    assert trend[10] == -1  # falling leg
    assert trend[-1] == 1  # reversed after the bottom
    # exactly one flip from -1 to +1 on clean V-shaped data
    flips = np.sum(np.diff(trend[1:]) != 0)
    assert flips == 1


def test_psar_acceleration_tightens_the_stop() -> None:
    high, low = _trend_bars(n=60, step=1.0)
    out = parabolic_sar(high, low, af_step=0.02, af_max=0.20)
    gap = (low - out["sar"]).iloc[5:]
    # the distance between price and stop must shrink as af accelerates
    assert gap.iloc[-1] < gap.iloc[0]


def test_psar_short_input_is_all_nan() -> None:
    out = parabolic_sar(pd.Series([100.0]), pd.Series([99.0]))
    assert out["sar"].isna().all()
    assert (out["trend"] == 0).all()


def test_psar_bad_params_raise() -> None:
    high, low = _trend_bars()
    with pytest.raises(ValueError, match="af_step"):
        parabolic_sar(high, low, af_step=0.0)
    with pytest.raises(ValueError, match="af_step"):
        parabolic_sar(high, low, af_step=0.5, af_max=0.2)
