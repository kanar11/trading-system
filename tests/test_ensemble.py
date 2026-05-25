"""Tests for the signal-ensemble combiners."""

import numpy as np
import pandas as pd
import pytest

from src.strategy.ensemble import majority_vote, weighted_sum, unanimous


def _sigframe(arrays: list[list[int]], cols: list[str] | None = None) -> pd.DataFrame:
    cols = cols or [f"s{i}" for i in range(len(arrays))]
    return pd.DataFrame(dict(zip(cols, arrays)))


# ---------------------------------------------------------------------------
# majority_vote
# ---------------------------------------------------------------------------

def test_majority_vote_basic():
    sf = _sigframe([[1, 1, -1], [1, -1, -1], [0, 1, -1]])
    out = majority_vote(sf)
    # rows: (1,1,0)=+2→+1  (1,-1,1)=+1→+1  (-1,-1,-1)=-3→-1
    assert out.tolist() == [1, 1, -1]


def test_majority_vote_tie_is_flat():
    # rows sum to zero — must collapse to flat
    sf = _sigframe([[1, -1], [-1, 1]])  # cols: s0=[1,-1] s1=[-1,1]
    out = majority_vote(sf)
    assert (out == 0).all()


def test_majority_vote_validates_signal_values():
    sf = _sigframe([[2, 1, 0]])
    with pytest.raises(ValueError, match="-1, 0, 1"):
        majority_vote(sf)


def test_majority_vote_rejects_empty():
    with pytest.raises(ValueError, match="empty"):
        majority_vote(pd.DataFrame())


# ---------------------------------------------------------------------------
# weighted_sum
# ---------------------------------------------------------------------------

def test_weighted_sum_with_dict_weights():
    sf = _sigframe([[1, 1, 1], [-1, -1, -1]], cols=["a", "b"])
    # weight a heavily — should override b
    out = weighted_sum(sf, weights={"a": 5.0, "b": 1.0})
    assert out.tolist() == [1, 1, 1]


def test_weighted_sum_with_list_weights():
    sf = _sigframe([[1, 1, 1], [-1, -1, -1]], cols=["a", "b"])
    out = weighted_sum(sf, weights=[1.0, 5.0])
    assert out.tolist() == [-1, -1, -1]


def test_weighted_sum_threshold_filters_weak_signal():
    sf = _sigframe([[1, 1, 1], [-1, 0, 0]], cols=["a", "b"])
    # equal weights: scores = (0, 0.5, 0.5)
    out_zero = weighted_sum(sf, threshold=0.0)
    out_thr = weighted_sum(sf, threshold=0.7)
    assert (out_thr == 0).all()
    assert (out_zero != 0).any()


def test_weighted_sum_rejects_wrong_length_weights():
    sf = _sigframe([[1, 0], [0, 1]])
    with pytest.raises(ValueError, match="length"):
        weighted_sum(sf, weights=[1.0])


def test_weighted_sum_rejects_zero_total_weight():
    sf = _sigframe([[1, 0], [0, 1]])
    with pytest.raises(ValueError, match="zero"):
        weighted_sum(sf, weights=[0.0, 0.0])


# ---------------------------------------------------------------------------
# unanimous
# ---------------------------------------------------------------------------

def test_unanimous_only_unanimous_rows_pass():
    sf = _sigframe([[1, 1, 0, -1, -1], [1, 1, 0, -1, 0]])
    out = unanimous(sf)
    # (1,1)=+1  (1,1)=+1  (0,0)=0  (-1,-1)=-1  (-1,0)=disagree → 0
    assert out.tolist() == [1, 1, 0, -1, 0]


def test_unanimous_with_single_column_is_identity():
    sf = _sigframe([[1, 0, -1, 1]])
    out = unanimous(sf)
    assert out.tolist() == [1, 0, -1, 1]
