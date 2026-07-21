"""Tests for fractional differentiation (FFD)."""

import numpy as np
import pandas as pd
import pytest

from src.data.fracdiff import ffd_weights, frac_diff_ffd


def _log_price(n: int = 500, seed: int = 3) -> pd.Series:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    log_price = np.log(100.0) + np.cumsum(rng.normal(0.0005, 0.01, n))
    return pd.Series(log_price, index=idx)


def test_weights_d_zero_is_identity_kernel() -> None:
    w = ffd_weights(0.0)
    assert np.array_equal(w, [1.0])


def test_weights_d_one_is_first_difference_kernel() -> None:
    w = ffd_weights(1.0)
    assert np.allclose(w, [1.0, -1.0])


def test_weights_recursion_matches_binomial() -> None:
    d = 0.4
    w = ffd_weights(d, threshold=1e-6)
    # w_k = -w_{k-1} (d-k+1)/k
    for k in range(1, len(w)):
        assert w[k] == pytest.approx(-w[k - 1] * (d - k + 1) / k)


def test_d_zero_returns_the_input() -> None:
    s = _log_price()
    out = frac_diff_ffd(s, d=0.0)
    assert out.name == "frac_diff"
    assert np.allclose(out.to_numpy(), s.to_numpy())


def test_d_one_matches_first_difference() -> None:
    s = _log_price()
    out = frac_diff_ffd(s, d=1.0)
    expected = s.diff()
    assert np.isnan(out.iloc[0])
    assert np.allclose(out.to_numpy()[1:], expected.to_numpy()[1:])


def test_warm_up_length_equals_kernel_width_minus_one() -> None:
    s = _log_price()
    w = ffd_weights(0.5, threshold=1e-3)
    out = frac_diff_ffd(s, d=0.5, threshold=1e-3)
    width = len(w)
    assert out.iloc[: width - 1].isna().all()
    assert out.iloc[width - 1 :].notna().all()


def test_fractional_diff_reduces_trend_memory() -> None:
    # a strongly trending (near unit-root) log-price series is highly
    # autocorrelated at lag 1; fractional differencing at d=0.4 cuts that
    s = _log_price()
    raw_autocorr = s.autocorr(lag=1)
    fd = frac_diff_ffd(s, d=0.4, threshold=1e-4).dropna()
    fd_autocorr = fd.autocorr(lag=1)
    assert raw_autocorr > 0.99
    assert abs(fd_autocorr) < raw_autocorr


def test_partial_d_keeps_more_memory_than_full_diff() -> None:
    # d=0.3 should retain more lag-1 autocorrelation than the full return
    s = _log_price()
    light = frac_diff_ffd(s, d=0.3, threshold=1e-4).dropna()
    full = frac_diff_ffd(s, d=1.0).dropna()
    assert abs(light.autocorr(lag=1)) > abs(full.autocorr(lag=1))


def test_bad_inputs_raise() -> None:
    s = _log_price(50)
    with pytest.raises(ValueError, match="d must be"):
        frac_diff_ffd(s, d=-0.5)
    with pytest.raises(ValueError, match="threshold"):
        ffd_weights(0.5, threshold=0.0)
    with pytest.raises(ValueError, match="max_width"):
        ffd_weights(0.5, max_width=0)
    with_nan = s.copy()
    with_nan.iloc[3] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        frac_diff_ffd(with_nan, d=0.5)
