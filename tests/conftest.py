"""Shared pytest fixtures for the test suite."""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_ohlcv() -> pd.DataFrame:
    """DataFrame with 250 days of synthetic OHLCV data.

    Prices follow a gentle uptrend with some noise.  Useful for any
    test that needs realistic-looking market data with high/low/close
    columns (e.g. regime detection, strategies, backtest engine).
    """
    np.random.seed(42)
    n = 250
    dates = pd.date_range("2020-01-01", periods=n, freq="B")

    # random walk with upward drift
    returns = np.random.normal(0.0004, 0.012, n)
    close = 100 * np.exp(np.cumsum(returns))

    high = close * (1 + np.abs(np.random.normal(0, 0.005, n)))
    low = close * (1 - np.abs(np.random.normal(0, 0.005, n)))
    open_ = close * (1 + np.random.normal(0, 0.003, n))
    volume = np.random.randint(1_000_000, 10_000_000, n)

    return pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        },
        index=dates,
    )


@pytest.fixture
def sample_prices(sample_ohlcv) -> pd.DataFrame:
    """DataFrame with close prices only (subset of sample_ohlcv)."""
    return sample_ohlcv[["close"]].copy()


@pytest.fixture
def sample_returns() -> pd.Series:
    """Series of mixed daily returns."""
    return pd.Series(
        [0.01, 0.02, -0.005, 0.015, -0.01, 0.008, -0.003, 0.012, 0.005, -0.007]
    )
