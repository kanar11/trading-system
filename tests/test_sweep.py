"""Tests for the parameter sweep runner."""

from pathlib import Path

import pandas as pd

from src.reporting.sweep import run_sweep


def test_sweep_with_injected_df(sample_ohlcv: pd.DataFrame, tmp_path: Path) -> None:
    result = run_sweep(
        ticker="TEST",
        lookbacks=[5, 10],
        thresholds=[0.0, 0.01],
        output_dir=tmp_path,
        df=sample_ohlcv,
    )
    # 2 lookbacks x 2 thresholds = 4 rows
    assert len(result) == 4
    assert (tmp_path / "sweep_results.csv").exists()


def test_sweep_results_sorted_by_sharpe(sample_ohlcv: pd.DataFrame, tmp_path: Path) -> None:
    result = run_sweep(
        ticker="TEST",
        lookbacks=[5, 10, 20],
        thresholds=[0.0, 0.01],
        output_dir=tmp_path,
        df=sample_ohlcv,
    )
    assert result["sharpe"].is_monotonic_decreasing


def test_sweep_expected_columns(sample_ohlcv: pd.DataFrame, tmp_path: Path) -> None:
    result = run_sweep(
        ticker="TEST",
        lookbacks=[5],
        thresholds=[0.0],
        output_dir=tmp_path,
        df=sample_ohlcv,
    )
    for col in ["lookback", "threshold", "sharpe", "num_trades", "win_rate"]:
        assert col in result.columns
