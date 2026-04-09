import pytest
import pandas as pd
import numpy as np
from src.reporting.metrics import calculate_metrics


def test_metrics_positive_returns():
    returns = pd.Series([0.01, 0.02, -0.005, 0.015, 0.01])
    metrics = calculate_metrics(returns)

    assert metrics["Total Return"] > 0
    assert metrics["Sharpe Ratio"] > 0
    assert metrics["Max Drawdown"] <= 0


def test_metrics_empty_returns():
    returns = pd.Series([], dtype=float)
    metrics = calculate_metrics(returns)

    assert metrics["Total Return"] == 0.0
    assert metrics["Sharpe Ratio"] == 0.0


def test_metrics_all_negative():
    returns = pd.Series([-0.01, -0.02, -0.015])
    metrics = calculate_metrics(returns)

    assert metrics["Total Return"] < 0
    assert metrics["Max Drawdown"] < 0