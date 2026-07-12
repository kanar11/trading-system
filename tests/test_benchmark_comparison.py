"""Tests for the strategy-vs-benchmark comparison table."""

import numpy as np
import pandas as pd
import pytest

from src.reporting.benchmark import benchmark_comparison


def _benchmark(n: int = 400, seed: int = 3) -> pd.Series:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2022-01-03", periods=n, freq="B")
    return pd.Series(rng.normal(0.0004, 0.011, n), index=idx)


def test_columns_and_rows_present() -> None:
    bench = _benchmark()
    table = benchmark_comparison(0.5 * bench, bench)
    assert list(table.columns) == ["strategy", "benchmark"]
    for row in ("ann_return", "ann_vol", "sharpe", "max_drawdown", "hit_rate", "beta"):
        assert row in table.index


def test_absolute_rows_filled_for_both_relative_only_for_strategy() -> None:
    bench = _benchmark()
    table = benchmark_comparison(0.5 * bench, bench)
    absolute = ["ann_return", "ann_vol", "sharpe", "max_drawdown", "hit_rate"]
    assert table.loc[absolute].notna().all().all()
    relative = ["beta", "tracking_error", "information_ratio", "correlation"]
    assert table.loc[relative, "benchmark"].isna().all()
    assert table.loc[relative, "strategy"].notna().all()


def test_half_beta_strategy_measures_correctly() -> None:
    bench = _benchmark()
    table = benchmark_comparison(0.5 * bench, bench)
    assert table.loc["beta", "strategy"] == pytest.approx(0.5, abs=1e-9)
    assert table.loc["ann_vol", "strategy"] == pytest.approx(
        0.5 * table.loc["ann_vol", "benchmark"]
    )
    assert table.loc["correlation", "strategy"] == pytest.approx(1.0)
    assert table.loc["up_capture", "strategy"] == pytest.approx(0.5, abs=1e-9)
    assert table.loc["down_capture", "strategy"] == pytest.approx(0.5, abs=1e-9)


def test_identical_series_relative_block_degenerates_cleanly() -> None:
    bench = _benchmark()
    table = benchmark_comparison(bench.copy(), bench)
    assert table.loc["beta", "strategy"] == pytest.approx(1.0, abs=1e-9)
    assert table.loc["tracking_error", "strategy"] == pytest.approx(0.0, abs=1e-12)
    assert np.isnan(table.loc["information_ratio", "strategy"])  # 0/0 guarded
    assert table.loc["ann_return", "strategy"] == pytest.approx(
        table.loc["ann_return", "benchmark"]
    )


def test_max_drawdown_is_negative_for_volatile_series() -> None:
    bench = _benchmark()
    table = benchmark_comparison(0.5 * bench, bench)
    assert table.loc["max_drawdown", "strategy"] < 0
    assert table.loc["max_drawdown", "benchmark"] < 0


def test_outperforming_strategy_has_positive_ir() -> None:
    bench = _benchmark()
    strat = bench + 0.0003  # steady daily outperformance
    table = benchmark_comparison(strat, bench)
    assert table.loc["information_ratio", "strategy"] > 1
    assert table.loc["jensen_alpha", "strategy"] > 0


def test_bad_inputs_raise() -> None:
    bench = _benchmark(50)
    with pytest.raises(ValueError, match="index"):
        benchmark_comparison(bench.iloc[:-1], bench)
    empty = pd.Series(dtype=float, index=pd.DatetimeIndex([]))
    with pytest.raises(ValueError, match="empty"):
        benchmark_comparison(empty, empty)
    with pytest.raises(ValueError, match="periods_per_year"):
        benchmark_comparison(bench, bench, periods_per_year=0)
