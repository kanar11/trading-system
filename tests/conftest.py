"""Shared pytest fixtures for the test suite."""

import pandas as pd
import pytest


@pytest.fixture
def sample_prices() -> pd.DataFrame:
    """DataFrame with 50 days of rising prices."""
    dates = pd.date_range("2020-01-01", periods=50, freq="B")
    prices = [100 + i * 0.5 for i in range(50)]
    return pd.DataFrame({"close": prices}, index=dates)


@pytest.fixture
def sample_returns() -> pd.Series:
    """Series of mixed daily returns."""
    return pd.Series(
        [0.01, 0.02, -0.005, 0.015, -0.01, 0.008, -0.003, 0.012, 0.005, -0.007]
    )
