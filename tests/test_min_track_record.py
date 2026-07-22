"""Tests for the Minimum Track Record Length statistic."""

import math

import numpy as np
import pandas as pd
import pytest

from src.validation.stat_tests import (
    minimum_track_record_length,
    probabilistic_sharpe_ratio,
)


def _returns(mean: float, vol: float = 0.01, n: int = 500, seed: int = 1) -> pd.Series:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2022-01-03", periods=n, freq="B")
    return pd.Series(rng.normal(mean, vol, n), index=idx)


def test_sample_longer_than_mintrl_clears_the_confidence() -> None:
    # the defining relationship: PSR(target) >= confidence  <=>  n >= MinTRL
    r = _returns(0.0015)
    mintrl = minimum_track_record_length(r, target_sharpe=0.0, confidence=0.95)
    assert math.isfinite(mintrl)
    assert len(r) >= mintrl
    assert probabilistic_sharpe_ratio(r, target_sharpe=0.0) >= 0.95


def test_sample_shorter_than_mintrl_falls_short() -> None:
    # truncate the same edge below MinTRL -> PSR must drop under confidence
    r = _returns(0.0015)
    mintrl = minimum_track_record_length(r, confidence=0.95)
    short = r.iloc[: int(mintrl) - 20]
    assert len(short) < mintrl
    assert probabilistic_sharpe_ratio(short, target_sharpe=0.0) < 0.95


def test_stronger_edge_needs_a_shorter_record() -> None:
    weak = minimum_track_record_length(_returns(0.0006))
    strong = minimum_track_record_length(_returns(0.0015))
    assert math.isfinite(weak)
    assert strong < weak


def test_higher_confidence_needs_a_longer_record() -> None:
    r = _returns(0.0015, seed=3)
    strict = minimum_track_record_length(r, confidence=0.99)
    loose = minimum_track_record_length(r, confidence=0.90)
    assert strict > loose


def test_higher_target_needs_a_longer_record() -> None:
    r = _returns(0.0015, seed=4)
    demanding = minimum_track_record_length(r, target_sharpe=0.5)
    easy = minimum_track_record_length(r, target_sharpe=0.0)
    assert demanding > easy


def test_edge_below_target_is_unprovable() -> None:
    flat = _returns(0.0, seed=5)
    assert minimum_track_record_length(flat, target_sharpe=1.0) == math.inf


def test_short_series_returns_inf() -> None:
    tiny = pd.Series([0.01, -0.005, 0.002])
    assert minimum_track_record_length(tiny) == math.inf


def test_bad_confidence_raises() -> None:
    r = _returns(0.0015, seed=6)
    with pytest.raises(ValueError, match="confidence"):
        minimum_track_record_length(r, confidence=0.0)
    with pytest.raises(ValueError, match="confidence"):
        minimum_track_record_length(r, confidence=1.0)
