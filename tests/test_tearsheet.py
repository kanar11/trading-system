"""Tests for the tear-sheet generator."""

import matplotlib

matplotlib.use("Agg")  # headless backend for CI

import numpy as np
import pandas as pd
import pytest
from matplotlib.figure import Figure

from src.reporting.tearsheet import generate_tearsheet


@pytest.fixture
def daily_returns():
    rng = np.random.default_rng(0)
    dates = pd.date_range("2021-01-01", periods=500, freq="B")
    return pd.Series(rng.normal(0.0005, 0.012, len(dates)), index=dates)


def test_returns_a_figure(daily_returns):
    fig = generate_tearsheet(daily_returns)
    assert isinstance(fig, Figure)
    # at least 6 subplots: equity, dd, rolling sharpe, heatmap, hist, table
    assert len(fig.axes) >= 6


def test_saves_to_disk(daily_returns, tmp_path):
    out = tmp_path / "report" / "tearsheet.png"
    fig = generate_tearsheet(daily_returns, output_path=out)
    assert out.exists()
    assert out.stat().st_size > 0
    assert isinstance(fig, Figure)


def test_with_benchmark_and_trade_log(daily_returns, tmp_path):
    rng = np.random.default_rng(1)
    benchmark = pd.Series(
        rng.normal(0.0003, 0.010, len(daily_returns)),
        index=daily_returns.index,
    )
    trade_log = pd.DataFrame({"trade_return": rng.normal(0.005, 0.02, 30), "direction": [1] * 30})
    out = tmp_path / "ts.png"
    fig = generate_tearsheet(
        daily_returns,
        benchmark=benchmark,
        trade_log=trade_log,
        output_path=out,
    )
    assert out.exists()
    assert isinstance(fig, Figure)


def test_empty_returns_raises():
    with pytest.raises(ValueError, match="empty"):
        generate_tearsheet(pd.Series(dtype=float))


def test_custom_rolling_window_runs(daily_returns):
    fig = generate_tearsheet(daily_returns, rolling_window=30)
    assert isinstance(fig, Figure)
