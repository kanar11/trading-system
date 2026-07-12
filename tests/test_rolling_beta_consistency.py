"""Consistency tests for the consolidated rolling-beta implementation.

Expansion #60 added ``reporting.attribution.rolling_alpha_beta`` without
noticing that ``risk.metrics.rolling_beta`` already computed the same
rolling OLS beta. The implementations are now consolidated — attribution
delegates its beta leg to risk.metrics — and these tests pin the contract
of the single shared implementation from both entry points.
"""

import numpy as np
import pandas as pd
import pytest

from src.reporting.attribution import rolling_alpha_beta
from src.risk.metrics import rolling_beta


def _series_pair(n: int = 250, seed: int = 14) -> tuple[pd.Series, pd.Series]:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2023-01-02", periods=n, freq="B")
    bench = pd.Series(rng.normal(0.0003, 0.011, n), index=idx)
    strat = 0.7 * bench + pd.Series(rng.normal(0.0, 0.003, n), index=idx)
    return strat, bench


def test_both_apis_return_identical_betas() -> None:
    strat, bench = _series_pair()
    direct = rolling_beta(strat, bench, window=63)
    via_attribution = rolling_alpha_beta(strat, bench, window=63)["beta"]
    # bitwise-identical: one implementation, two entry points
    assert np.array_equal(direct.to_numpy(), via_attribution.to_numpy(), equal_nan=True)


def test_alpha_leg_still_recovers_known_coefficients() -> None:
    strat, bench = _series_pair()
    exact = 0.5 * bench + 0.0002
    out = rolling_alpha_beta(exact, bench, window=40)
    tail = out.iloc[40:]
    assert np.allclose(tail["beta"].to_numpy(), 0.5)
    assert np.allclose(tail["alpha"].to_numpy(), 0.0002 * 252)


def test_risk_entry_point_keeps_inner_join_semantics() -> None:
    # rolling_beta accepts differing indexes and aligns on the intersection —
    # a contract the strict attribution wrapper must not have changed
    strat, bench = _series_pair()
    shifted = strat.iloc[20:]
    out = rolling_beta(shifted, bench, window=30)
    assert len(out) == len(shifted)
    assert out.iloc[30:].notna().all()


def test_attribution_entry_point_stays_strict_about_indexes() -> None:
    strat, bench = _series_pair()
    with pytest.raises(ValueError, match="index"):
        rolling_alpha_beta(strat.iloc[:-1], bench)


def test_constant_benchmark_gives_nan_from_both_apis() -> None:
    idx = pd.date_range("2024-01-01", periods=30, freq="B")
    bench = pd.Series(0.015625, index=idx)  # binary-exact -> variance exactly 0
    strat = pd.Series(np.linspace(0.0, 0.01, 30), index=idx)
    assert rolling_beta(strat, bench, window=10).iloc[10:].isna().all()
    assert rolling_alpha_beta(strat, bench, window=10)["beta"].iloc[10:].isna().all()


def test_window_parameter_is_forwarded() -> None:
    strat, bench = _series_pair()
    short = rolling_alpha_beta(strat, bench, window=20)["beta"]
    long = rolling_alpha_beta(strat, bench, window=120)["beta"]
    # warm-up lengths follow the window, proving the forward
    assert short.iloc[19:].notna().all()
    assert long.iloc[:119].isna().all()
    assert long.iloc[119:].notna().all()
