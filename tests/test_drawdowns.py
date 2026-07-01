"""Tests for the worst-drawdown episode table."""

import pandas as pd
import pytest

from src.reporting.drawdowns import drawdown_table


def _series(values: list[float]) -> pd.Series:
    idx = pd.date_range("2020-01-01", periods=len(values), freq="B")
    return pd.Series(values, index=idx)


def test_single_recovered_episode() -> None:
    r = _series([-0.2, 0.1, 0.25, -0.1, 0.15])
    table = drawdown_table(r)
    assert len(table) == 1
    row = table.iloc[0]
    assert row["depth"] == pytest.approx(-0.1)
    assert row["peak_date"] == r.index[2]
    assert row["trough_date"] == r.index[3]
    assert row["recovery_date"] == r.index[4]
    assert row["length"] == 2


def test_unrecovered_episode_has_nat_recovery() -> None:
    table = drawdown_table(_series([0.1, -0.3]))
    assert len(table) == 1
    assert table.iloc[0]["depth"] == pytest.approx(-0.3)
    assert pd.isna(table.iloc[0]["recovery_date"])


def test_episodes_sorted_deepest_first() -> None:
    r = _series([0.1, -0.05, 0.1, -0.2, 0.3])
    table = drawdown_table(r)
    assert len(table) == 2
    assert table.iloc[0]["depth"] == pytest.approx(-0.2)
    assert table.iloc[1]["depth"] == pytest.approx(-0.05)


def test_top_n_limits_rows() -> None:
    r = _series([0.1, -0.05, 0.1, -0.2, 0.3])
    table = drawdown_table(r, top_n=1)
    assert len(table) == 1
    assert table.iloc[0]["depth"] == pytest.approx(-0.2)


def test_empty_returns_empty_table_with_columns() -> None:
    table = drawdown_table(pd.Series([], dtype=float))
    assert table.empty
    assert list(table.columns) == [
        "peak_date",
        "trough_date",
        "recovery_date",
        "depth",
        "length",
    ]


def test_monotonic_up_has_no_drawdowns() -> None:
    assert drawdown_table(_series([0.01, 0.02, 0.03])).empty
