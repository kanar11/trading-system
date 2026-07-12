"""Tests for Treynor-Mazuy and Henriksson-Merton timing regressions."""

import numpy as np
import pandas as pd
import pytest

from src.validation import henriksson_merton, treynor_mazuy


def _benchmark(n: int = 500, seed: int = 2) -> pd.Series:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2021-01-04", periods=n, freq="B")
    return pd.Series(rng.normal(0.0002, 0.012, n), index=idx)


def test_linear_strategy_has_no_timing() -> None:
    # noise keeps the regression well-posed: an EXACT linear fit leaves only
    # float-dust residuals and the t-stat becomes a ratio of two dusts
    bench = _benchmark()
    rng = np.random.default_rng(7)
    noise = pd.Series(rng.normal(0, 0.001, len(bench)), index=bench.index)
    strat = 0.5 * bench + 0.0001 + noise
    result = treynor_mazuy(strat, bench)
    assert result.beta == pytest.approx(0.5, abs=0.02)
    assert abs(result.gamma_tstat) < 2  # timing coefficient insignificant
    assert abs(result.alpha) < 5e-4
    assert result.n_obs == 500


def test_treynor_mazuy_recovers_quadratic_convexity() -> None:
    bench = _benchmark()
    strat = bench + 0.5 * bench**2
    result = treynor_mazuy(strat, bench)
    assert result.gamma == pytest.approx(0.5, abs=1e-9)
    assert result.beta == pytest.approx(1.0, abs=1e-9)
    assert result.gamma_tstat > 3  # exact fit -> hugely significant
    assert result.r_squared == pytest.approx(1.0)


def test_henriksson_merton_recovers_option_payoff() -> None:
    bench = _benchmark()
    strat = bench + 0.3 * np.maximum(bench, 0.0)
    result = henriksson_merton(strat, bench)
    assert result.gamma == pytest.approx(0.3, abs=1e-9)
    assert result.beta == pytest.approx(1.0, abs=1e-9)
    assert result.gamma_tstat > 3


def test_hm_detects_tm_style_convexity_directionally() -> None:
    bench = _benchmark()
    rng = np.random.default_rng(5)
    noise = pd.Series(rng.normal(0, 0.001, len(bench)), index=bench.index)
    strat = bench + 0.8 * bench**2 + noise
    result = henriksson_merton(strat, bench)
    assert result.gamma > 0  # convex payoff shows up as positive HM gamma


def test_negative_timing_is_detected() -> None:
    bench = _benchmark()
    strat = bench - 0.4 * bench**2  # concave: pays for anti-timing
    result = treynor_mazuy(strat, bench)
    assert result.gamma == pytest.approx(-0.4, abs=1e-9)
    assert result.gamma_tstat < -3


def test_pure_benchmark_is_alpha_and_gamma_free() -> None:
    bench = _benchmark()
    result = treynor_mazuy(bench, bench)
    assert result.beta == pytest.approx(1.0, abs=1e-9)
    assert result.alpha == pytest.approx(0.0, abs=1e-12)
    assert result.gamma == pytest.approx(0.0, abs=1e-9)


def test_bad_inputs_raise() -> None:
    bench = _benchmark(50)
    with pytest.raises(ValueError, match="index"):
        treynor_mazuy(bench.iloc[:-1], bench)
    short = bench.iloc[:5]
    with pytest.raises(ValueError, match="observations"):
        henriksson_merton(short, short)
