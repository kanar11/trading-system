"""Tests for the multi-strategy league table."""

import numpy as np
import pandas as pd
import pytest

from src.reporting.league import strategy_league


def _series(mean: float, vol: float, n: int = 300, seed: int = 0) -> pd.Series:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2022-01-03", periods=n, freq="B")
    return pd.Series(rng.normal(mean, vol, n), index=idx)


def test_table_shape_and_default_sharpe_ranking() -> None:
    table = strategy_league(
        {
            "strong": _series(0.001, 0.005, seed=1),
            "weak": _series(0.0, 0.01, seed=2),
            "wild": _series(0.0005, 0.03, seed=3),
        }
    )
    assert table.index.name == "strategy"
    assert list(table.index)[0] == "strong"  # highest Sharpe first
    for column in ("ann_return", "ann_vol", "sharpe", "max_drawdown", "hit_rate", "n_obs"):
        assert column in table.columns
    assert "beta" not in table.columns  # no benchmark supplied


def test_benchmark_adds_relative_columns() -> None:
    bench = _series(0.0004, 0.01, seed=4)
    table = strategy_league({"tracker": 0.5 * bench, "alpha": bench + 0.0002}, benchmark=bench)
    assert table.loc["tracker", "beta"] == pytest.approx(0.5, abs=1e-9)
    assert table.loc["alpha", "beta"] == pytest.approx(1.0, abs=1e-9)
    assert table.loc["alpha", "information_ratio"] > 1


def test_zero_tracking_error_gives_nan_ir() -> None:
    bench = _series(0.0004, 0.01, seed=5)
    table = strategy_league({"clone": bench.copy()}, benchmark=bench)
    assert np.isnan(table.loc["clone", "information_ratio"])


def test_sort_by_other_columns() -> None:
    table = strategy_league(
        {"calm": _series(0.0002, 0.004, seed=6), "wild": _series(0.0002, 0.03, seed=7)},
        sort_by="ann_vol",
    )
    assert list(table.index)[0] == "wild"


def test_mixed_lengths_report_their_own_n_obs() -> None:
    long = _series(0.0005, 0.01, n=400, seed=8)
    short = _series(0.0005, 0.01, n=100, seed=9)
    table = strategy_league({"long": long, "short": short})
    assert table.loc["long", "n_obs"] == 400
    assert table.loc["short", "n_obs"] == 100


def test_bad_inputs_raise() -> None:
    with pytest.raises(ValueError, match="empty"):
        strategy_league({})
    with pytest.raises(ValueError, match="empty"):
        strategy_league({"x": pd.Series(dtype=float)})
    good = {"x": _series(0.0, 0.01, seed=10)}
    with pytest.raises(ValueError, match="sort_by"):
        strategy_league(good, sort_by="nonsense")
    with pytest.raises(ValueError, match="periods_per_year"):
        strategy_league(good, periods_per_year=0)
