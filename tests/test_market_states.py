"""Tests for bull/bear market-state labels."""

import numpy as np
import pandas as pd
import pytest

from src.regime import BEAR, BULL, bull_bear_labels, regime_performance


def _series(values: list[float]) -> pd.Series:
    idx = pd.date_range("2024-01-01", periods=len(values), freq="B")
    return pd.Series(values, index=idx)


def test_monotone_rally_is_all_bull() -> None:
    close = _series(list(np.linspace(100, 200, 50)))
    labels = bull_bear_labels(close)
    assert labels.name == "market_state"
    assert (labels == BULL).all()


def test_bear_starts_at_the_threshold_cross() -> None:
    close = _series([100.0, 95.0, 90.0, 85.0, 79.0, 75.0])
    labels = bull_bear_labels(close, threshold=0.20)
    # 85 is a 15% drawdown (still bull); 79 crosses the 20% line
    assert list(labels) == [BULL, BULL, BULL, BULL, BEAR, BEAR]


def test_bull_resumes_after_threshold_rally_off_the_trough() -> None:
    close = _series([100.0, 79.0, 70.0, 75.0, 85.0, 90.0])
    labels = bull_bear_labels(close, threshold=0.20)
    # trough 70; 84 = 70 * 1.2 -> the 85 bar flips back to bull
    assert list(labels) == [BULL, BEAR, BEAR, BEAR, BULL, BULL]


def test_shallow_corrections_never_flip() -> None:
    rng = np.random.default_rng(11)
    wobble = 100.0 + np.cumsum(rng.normal(0.05, 0.5, 300))
    close = _series(list(np.maximum(wobble, 90.0)))  # bounded 10% dips
    labels = bull_bear_labels(close, threshold=0.20)
    assert (labels == BULL).all()


def test_peak_resets_after_recovery() -> None:
    # crash from 100, recover to 90 (bull again), then a 20% fall from the
    # NEW peak 90 (to 72) is needed to re-enter bear — 75 is not enough
    close = _series([100.0, 79.0, 70.0, 90.0, 75.0, 71.0])
    labels = bull_bear_labels(close, threshold=0.20)
    assert list(labels) == [BULL, BEAR, BEAR, BULL, BULL, BEAR]


def test_feeds_regime_performance() -> None:
    down = list(np.linspace(100, 60, 40))
    up = list(np.linspace(60, 130, 60))
    close = _series(down + up)
    labels = bull_bear_labels(close, threshold=0.20)
    returns = close.pct_change().fillna(0.0)
    table = regime_performance(returns, labels)
    assert set(table.index) == {BULL, BEAR}
    assert table.loc[BEAR, "ann_return"] < 0 < table.loc[BULL, "ann_return"]


def test_bad_inputs_raise() -> None:
    close = _series([100.0, 90.0])
    with pytest.raises(ValueError, match="threshold"):
        bull_bear_labels(close, threshold=0.0)
    with pytest.raises(ValueError, match="threshold"):
        bull_bear_labels(close, threshold=1.0)
    with pytest.raises(ValueError, match="positive"):
        bull_bear_labels(_series([100.0, -1.0]))
    empty = pd.Series(dtype=float, index=pd.DatetimeIndex([]))
    with pytest.raises(ValueError, match="empty"):
        bull_bear_labels(empty)
