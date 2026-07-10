"""Tests for rolling alpha/beta attribution."""

import numpy as np
import pandas as pd
import pytest

from src.reporting.attribution import rolling_alpha_beta


def _benchmark(n: int = 300, seed: int = 8) -> pd.Series:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2022-01-03", periods=n, freq="B")
    return pd.Series(rng.normal(0.0004, 0.01, n), index=idx)


def test_recovers_constant_beta_and_alpha() -> None:
    bench = _benchmark()
    daily_alpha = 0.0002
    strat = 0.5 * bench + daily_alpha
    out = rolling_alpha_beta(strat, bench, window=63)
    assert list(out.columns) == ["alpha", "beta"]
    tail = out.iloc[63:]
    assert np.allclose(tail["beta"].to_numpy(), 0.5)
    assert np.allclose(tail["alpha"].to_numpy(), daily_alpha * 252)


def test_tracking_the_benchmark_gives_beta_one_alpha_zero() -> None:
    bench = _benchmark()
    out = rolling_alpha_beta(bench, bench, window=40)
    tail = out.iloc[40:]
    assert np.allclose(tail["beta"].to_numpy(), 1.0)
    assert np.allclose(tail["alpha"].to_numpy(), 0.0, atol=1e-10)


def test_warm_up_is_nan() -> None:
    bench = _benchmark(80)
    out = rolling_alpha_beta(0.7 * bench, bench, window=30)
    assert out.iloc[: 30 - 1].isna().all().all()
    assert out.iloc[30:].notna().all().all()


def test_matches_direct_window_computation() -> None:
    bench = _benchmark(120, seed=3)
    rng = np.random.default_rng(4)
    strat = 0.8 * bench + pd.Series(rng.normal(0, 0.002, 120), index=bench.index)
    window = 50
    out = rolling_alpha_beta(strat, bench, window=window)
    # check one interior bar against the explicit trailing-window OLS
    t = 90
    s_win = strat.iloc[t - window + 1 : t + 1]
    b_win = bench.iloc[t - window + 1 : t + 1]
    expected_beta = float(np.cov(s_win, b_win, ddof=1)[0, 1] / np.var(b_win, ddof=1))
    assert out["beta"].iloc[t] == pytest.approx(expected_beta)
    expected_alpha = (float(s_win.mean()) - expected_beta * float(b_win.mean())) * 252
    assert out["alpha"].iloc[t] == pytest.approx(expected_alpha)


def test_constant_benchmark_window_is_nan() -> None:
    # binary-exact constant benchmark -> rolling variance is exactly 0
    idx = pd.date_range("2024-01-01", periods=20, freq="B")
    bench = pd.Series(0.015625, index=idx)
    strat = pd.Series(np.linspace(0.0, 0.01, 20), index=idx)
    out = rolling_alpha_beta(strat, bench, window=5)
    assert out["beta"].iloc[5:].isna().all()


def test_bad_inputs_raise() -> None:
    bench = _benchmark(50)
    with pytest.raises(ValueError, match="index"):
        rolling_alpha_beta(bench.iloc[:-1], bench)
    with pytest.raises(ValueError, match="window"):
        rolling_alpha_beta(bench, bench, window=1)
    with pytest.raises(ValueError, match="periods_per_year"):
        rolling_alpha_beta(bench, bench, periods_per_year=0)
