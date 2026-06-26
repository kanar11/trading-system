"""Tests for equity-curve plotting."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import pytest
from matplotlib.figure import Figure

from quantbt.reporting.plots import plot_equity


def _equity_df(n: int = 50) -> pd.DataFrame:
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    return pd.DataFrame({"equity_curve": pd.Series(range(1, n + 1)) / n + 1}, index=dates)


def test_returns_figure() -> None:
    fig = plot_equity(_equity_df())
    assert isinstance(fig, Figure)
    plt.close(fig)


def test_missing_equity_column_raises() -> None:
    with pytest.raises(ValueError, match="equity_curve"):
        plot_equity(pd.DataFrame({"close": [1.0, 2.0]}))


def test_saves_file_and_creates_dirs(tmp_path: Path) -> None:
    target = tmp_path / "nested" / "equity.png"
    fig = plot_equity(_equity_df(), save_path=target)
    assert target.exists()
    plt.close(fig)


def test_benchmark_overlay_adds_legend() -> None:
    df = _equity_df()
    benchmark = df["equity_curve"] * 0.9
    fig = plot_equity(df, benchmark=benchmark)
    ax = fig.axes[0]
    assert ax.get_legend() is not None
    plt.close(fig)
