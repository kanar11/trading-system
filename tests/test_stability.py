"""Tests for subperiod performance-stability analysis."""

import numpy as np
import pandas as pd
import pytest

from src.validation import stability_score, subperiod_stats


def _series(values: np.ndarray) -> pd.Series:
    idx = pd.date_range("2020-01-01", periods=len(values), freq="B")
    return pd.Series(values, index=idx)


def test_table_shape_and_columns() -> None:
    rng = np.random.default_rng(1)
    table = subperiod_stats(_series(rng.normal(0.0005, 0.01, 400)), n_periods=4)
    assert list(table.index) == ["period_0", "period_1", "period_2", "period_3"]
    for column in (
        "n_obs",
        "total_return",
        "ann_return",
        "ann_vol",
        "sharpe",
        "max_drawdown",
        "hit_rate",
    ):
        assert column in table.columns
    assert int(table["n_obs"].sum()) == 400


def test_steady_strategy_scores_consistent() -> None:
    # a genuinely steady edge: a drift that dominates the noise in every
    # window, so all five subperiods are clearly profitable
    rng = np.random.default_rng(2)
    steady = _series(rng.normal(0.001, 0.002, 500))
    score = stability_score(steady, n_periods=5)
    assert score["positive_fraction"] == 1.0
    assert score["sharpe_consistency"] > 1.5  # low dispersion of per-window Sharpe
    assert score["worst_sharpe"] > 0


def test_concentrated_edge_scores_worse_than_steady() -> None:
    rng = np.random.default_rng(3)
    n = 500
    # all the gains happen in the first fifth; the rest is zero-mean noise
    concentrated = rng.normal(0.0, 0.002, n)
    concentrated[:100] += 0.001
    steady = rng.normal(0.001, 0.002, n)
    conc_score = stability_score(_series(concentrated), n_periods=5)
    steady_score = stability_score(_series(steady), n_periods=5)
    assert conc_score["positive_fraction"] < steady_score["positive_fraction"]
    assert conc_score["worst_sharpe"] < steady_score["worst_sharpe"]


def test_per_window_stats_match_direct_computation() -> None:
    values = np.array([0.01, -0.02, 0.03, -0.01, 0.02, 0.0])
    table = subperiod_stats(_series(values), n_periods=2, periods_per_year=252)
    first = values[:3]
    expected_ann_return = float(first.mean()) * 252
    assert table.loc["period_0", "ann_return"] == pytest.approx(expected_ann_return)
    expected_total = float(np.prod(1 + first) - 1)
    assert table.loc["period_0", "total_return"] == pytest.approx(expected_total)


def test_uneven_split_distributes_the_remainder() -> None:
    # 10 bars into 3 windows -> 4, 3, 3
    table = subperiod_stats(_series(np.full(10, 0.001)), n_periods=3)
    assert list(table["n_obs"]) == [4.0, 3.0, 3.0]


def test_single_bar_window_has_nan_sharpe() -> None:
    # 2 bars, 2 windows -> each window is a single bar (std undefined)
    table = subperiod_stats(_series(np.array([0.01, -0.01])), n_periods=2)
    assert np.isnan(table["sharpe"]).all()
    score = stability_score(_series(np.array([0.01, -0.01])), n_periods=2)
    assert np.isnan(score["sharpe_consistency"])


def test_bad_inputs_raise() -> None:
    s = _series(np.full(20, 0.001))
    with pytest.raises(ValueError, match="empty"):
        subperiod_stats(_series(np.array([])))
    with pytest.raises(ValueError, match="n_periods"):
        subperiod_stats(s, n_periods=1)
    with pytest.raises(ValueError, match="n_periods"):
        subperiod_stats(s, n_periods=21)
    with pytest.raises(ValueError, match="periods_per_year"):
        subperiod_stats(s, periods_per_year=0)
