"""Tests for the self-contained HTML performance report."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.reporting.html_report import html_report, save_html_report


def _returns(n: int = 300, seed: int = 6) -> pd.Series:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2022-01-03", periods=n, freq="B")
    return pd.Series(rng.normal(0.0005, 0.01, n), index=idx)


def test_document_structure_and_sections() -> None:
    doc = html_report(_returns(), title="TSMOM on SPY")
    assert doc.startswith("<!DOCTYPE html>")
    assert doc.rstrip().endswith("</html>")
    assert "<h1>TSMOM on SPY</h1>" in doc
    assert "Summary statistics" in doc
    assert "Monthly returns" in doc
    assert "Top 5 drawdowns" in doc
    assert "<script" not in doc  # self-contained, no scripts


def test_without_benchmark_only_strategy_column() -> None:
    doc = html_report(_returns())
    assert "Benchmark: none" in doc
    assert "ann_return" in doc
    assert "tracking_error" not in doc  # relative block dropped


def test_with_benchmark_includes_relative_block() -> None:
    returns = _returns()
    bench = _returns(seed=9)
    doc = html_report(returns, benchmark=bench)
    assert "Benchmark: included" in doc
    assert "tracking_error" in doc
    assert "information_ratio" in doc


def test_report_contains_actual_numbers() -> None:
    returns = _returns()
    doc = html_report(returns)
    ann_vol = float(returns.std(ddof=1)) * np.sqrt(252)
    assert f"{ann_vol:.4f}" in doc


def test_date_range_in_metadata() -> None:
    doc = html_report(_returns())
    assert "2022-01-03" in doc
    assert "300 bars" in doc


def test_save_writes_utf8_file(tmp_path: Path) -> None:
    target = save_html_report(tmp_path / "report.html", _returns(), title="Zażółć")
    assert target.exists()
    content = target.read_text(encoding="utf-8")
    assert "Zażółć" in content
    assert content.startswith("<!DOCTYPE html>")


def test_top_drawdowns_parameter() -> None:
    doc = html_report(_returns(), top_drawdowns=3)
    assert "Top 3 drawdowns" in doc


def test_empty_returns_raise() -> None:
    empty = pd.Series(dtype=float, index=pd.DatetimeIndex([]))
    with pytest.raises(ValueError, match="empty"):
        html_report(empty)
