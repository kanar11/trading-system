"""Tests for equity / drawdown curve transforms."""

import pandas as pd
import pytest

from src.backtest.curves import drawdown_series, equity_curve


def test_equity_curve_compounds() -> None:
    eq = equity_curve(pd.Series([0.1, -0.1, 0.05]))
    assert list(eq) == pytest.approx([1.1, 0.99, 1.0395])


def test_equity_curve_initial_scales() -> None:
    eq = equity_curve(pd.Series([0.1, -0.1]), initial=1000.0)
    assert list(eq) == pytest.approx([1100.0, 990.0])


def test_equity_curve_fills_na() -> None:
    eq = equity_curve(pd.Series([0.1, None, 0.1]))
    assert list(eq) == pytest.approx([1.1, 1.1, 1.21])


def test_equity_curve_empty() -> None:
    assert equity_curve(pd.Series([], dtype=float)).empty


def test_drawdown_zero_when_monotonic() -> None:
    dd = drawdown_series(pd.Series([0.01] * 5))
    assert (dd <= 1e-12).all()
    assert dd.iloc[-1] == pytest.approx(0.0)


def test_drawdown_trough() -> None:
    dd = drawdown_series(pd.Series([0.1, -0.2, 0.05]))
    # equity 1.1, 0.88, 0.924 -> peak 1.1 -> dd 0, -0.2, -0.16
    assert dd.iloc[0] == pytest.approx(0.0)
    assert dd.min() == pytest.approx(-0.2)
    assert dd.iloc[-1] == pytest.approx(0.88 * 1.05 / 1.1 - 1)


def test_drawdown_empty() -> None:
    assert drawdown_series(pd.Series([], dtype=float)).empty
