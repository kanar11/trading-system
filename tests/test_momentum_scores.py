"""Tests for the 12-1 momentum score and 52-week-high distance."""

import numpy as np
import pandas as pd
import pytest

from src.indicators import distance_from_high, lookback_return


def _line(n: int = 300, start: float = 100.0, step: float = 0.5) -> pd.Series:
    return pd.Series(start + step * np.arange(n, dtype=float))


def test_lookback_return_skip_convention() -> None:
    close = _line()
    out = lookback_return(close, lookback=252, skip=21)
    # score_t = close[t-21] / close[t-252] - 1, checked explicitly
    t = 280
    expected = close.iloc[t - 21] / close.iloc[t - 252] - 1
    assert out.iloc[t] == pytest.approx(expected)
    assert out.iloc[: 252 - 1].isna().all()  # warm-up


def test_lookback_return_skip_zero_is_plain_return() -> None:
    close = _line(60)
    out = lookback_return(close, lookback=20, skip=0)
    expected = close / close.shift(20) - 1
    assert np.allclose(out.iloc[20:].to_numpy(), expected.iloc[20:].to_numpy())


def test_lookback_return_positive_in_uptrend_negative_in_downtrend() -> None:
    up = lookback_return(_line(step=0.5), lookback=100, skip=10)
    down = lookback_return(_line(step=-0.2, start=200.0), lookback=100, skip=10)
    assert (up.iloc[100:] > 0).all()
    assert (down.iloc[100:] < 0).all()


def test_lookback_return_bad_params_raise() -> None:
    close = _line(50)
    with pytest.raises(ValueError, match="skip"):
        lookback_return(close, skip=-1)
    with pytest.raises(ValueError, match="lookback"):
        lookback_return(close, lookback=21, skip=21)


def test_distance_from_high_is_zero_at_fresh_highs() -> None:
    close = _line(120)  # strictly rising -> every bar is the rolling high
    out = distance_from_high(close, window=50)
    assert np.allclose(out.iloc[49:].to_numpy(), 0.0)
    assert out.iloc[:48].isna().all()


def test_distance_from_high_measures_the_drawdown_from_window_high() -> None:
    values = [100.0] * 10 + [80.0] * 10
    close = pd.Series(values)
    out = distance_from_high(close, window=20)
    assert out.iloc[-1] == pytest.approx(80.0 / 100.0 - 1)  # 20% below the high


def test_distance_from_high_never_positive() -> None:
    rng = np.random.default_rng(6)
    close = pd.Series(100.0 * np.cumprod(1 + rng.normal(0, 0.01, 300)))
    out = distance_from_high(close, window=60)
    assert (out.dropna() <= 1e-12).all()


def test_distance_from_high_bad_window_raises() -> None:
    with pytest.raises(ValueError, match="window"):
        distance_from_high(_line(20), window=0)
