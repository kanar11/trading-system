"""Tests for the HMM regime-switching strategy."""

import numpy as np
import pandas as pd
import pytest

from src.strategy.hmm_regime import hmm_regime_strategy


def _bull_then_bear(n: int = 150, seed: int = 0) -> tuple[pd.DataFrame, slice, slice]:
    """Price built from a calm high-mean regime then a turbulent low-mean one."""
    rng = np.random.default_rng(seed)
    bull = rng.normal(0.002, 0.005, n)
    bear = rng.normal(-0.003, 0.02, n)
    rets = np.concatenate([bull, bear])
    close = 100.0 * np.cumprod(1.0 + rets)
    idx = pd.date_range("2020-01-01", periods=2 * n, freq="B")
    df = pd.DataFrame({"close": close}, index=idx)
    return df, slice(0, n), slice(n, 2 * n)


def test_outputs_columns_and_signal_domain() -> None:
    df, _, _ = _bull_then_bear()
    out = hmm_regime_strategy(df, n_states=2)
    assert "hmm_state" in out.columns
    assert "signal" in out.columns
    assert set(out["signal"].unique()).issubset({-1, 0, 1})


def test_trades_with_the_regime() -> None:
    df, bull, bear = _bull_then_bear()
    out = hmm_regime_strategy(df, n_states=2)
    # long dominates the bull leg, short dominates the bear leg
    assert (out["signal"].iloc[bull] == 1).mean() > 0.7
    assert (out["signal"].iloc[bear] == -1).mean() > 0.7


def test_allow_short_false_has_no_shorts() -> None:
    df, _, _ = _bull_then_bear()
    out = hmm_regime_strategy(df, n_states=2, allow_short=False)
    assert (out["signal"] >= 0).all()


def test_deterministic() -> None:
    df, _, _ = _bull_then_bear()
    a = hmm_regime_strategy(df, n_states=2)
    b = hmm_regime_strategy(df, n_states=2)
    assert a["signal"].equals(b["signal"])


def test_does_not_mutate_input() -> None:
    df, _, _ = _bull_then_bear()
    before = set(df.columns)
    hmm_regime_strategy(df, n_states=2)
    assert set(df.columns) == before


def test_missing_close_raises() -> None:
    with pytest.raises(ValueError, match="close"):
        hmm_regime_strategy(pd.DataFrame({"open": [1.0, 2.0, 3.0]}))


def test_bad_n_states_raises() -> None:
    df, _, _ = _bull_then_bear()
    with pytest.raises(ValueError, match="n_states"):
        hmm_regime_strategy(df, n_states=1)
