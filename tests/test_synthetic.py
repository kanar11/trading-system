"""Tests for the synthetic GBM OHLCV generator."""

import numpy as np
import pandas as pd
import pytest

from src.data.quality import check_ohlcv
from src.data.synthetic import generate_gbm_ohlcv


def test_shape_columns_and_index() -> None:
    df = generate_gbm_ohlcv(n_days=120, seed=0)
    assert df.shape == (120, 5)
    assert list(df.columns) == ["open", "high", "low", "close", "volume"]
    assert isinstance(df.index, pd.DatetimeIndex)


def test_output_passes_quality_audit() -> None:
    df = generate_gbm_ohlcv(n_days=250, seed=1)
    report = check_ohlcv(df)
    assert report.non_positive_prices == 0
    assert report.ohlc_inconsistencies == 0
    assert report.duplicate_timestamps == 0
    assert report.unsorted_index is False


def test_prices_are_positive() -> None:
    df = generate_gbm_ohlcv(n_days=200, seed=2)
    assert (df[["open", "high", "low", "close"]] > 0).all().all()
    assert (df["volume"] > 0).all()


def test_reproducible_with_seed() -> None:
    a = generate_gbm_ohlcv(n_days=100, seed=42)
    b = generate_gbm_ohlcv(n_days=100, seed=42)
    pd.testing.assert_frame_equal(a, b)


def test_different_seeds_differ() -> None:
    a = generate_gbm_ohlcv(n_days=100, seed=1)
    b = generate_gbm_ohlcv(n_days=100, seed=2)
    assert not np.allclose(a["close"].to_numpy(), b["close"].to_numpy())


def test_zero_vol_is_deterministic_drift() -> None:
    df = generate_gbm_ohlcv(n_days=100, mu=0.2, sigma=0.0, seed=0)
    # no diffusion + positive drift -> strictly increasing close
    assert (df["close"].diff().dropna() > 0).all()


def test_validation_errors() -> None:
    with pytest.raises(ValueError, match="n_days"):
        generate_gbm_ohlcv(n_days=0)
    with pytest.raises(ValueError, match="start_price"):
        generate_gbm_ohlcv(start_price=0)
    with pytest.raises(ValueError, match="sigma"):
        generate_gbm_ohlcv(sigma=-0.1)
