"""Tests for the Sharpe-ratio heatmap plotter."""

from pathlib import Path

import pandas as pd
import pytest

from plot_heatmap import plot_heatmap


def _write_sweep_csv(path: Path) -> None:
    df = pd.DataFrame(
        {
            "lookback": [5, 5, 10, 10],
            "threshold": [0.0, 0.01, 0.0, 0.01],
            "sharpe": [0.5, 0.8, 1.1, 0.3],
        }
    )
    df.to_csv(path, index=False)


def test_missing_file_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)  # avoid finding a real results/ fallback CSV
    with pytest.raises(FileNotFoundError):
        plot_heatmap(str(tmp_path / "does_not_exist.csv"))


def test_missing_columns_raises(tmp_path: Path) -> None:
    csv = tmp_path / "bad.csv"
    pd.DataFrame({"foo": [1], "bar": [2]}).to_csv(csv, index=False)
    with pytest.raises(ValueError, match="missing required columns"):
        plot_heatmap(str(csv))


def test_creates_heatmap_png(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    csv = tmp_path / "sweep_results.csv"
    _write_sweep_csv(csv)
    monkeypatch.chdir(tmp_path)

    plot_heatmap(str(csv))

    assert (tmp_path / "results" / "parameter_heatmap.png").exists()
