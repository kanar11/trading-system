"""Tests for post-trade transaction-cost analysis (TCA)."""

import pytest

from src.execution.tca import execution_vwap, implementation_shortfall, vwap_slippage


def test_execution_vwap_weighted() -> None:
    assert execution_vwap([100.0, 102.0], [10.0, 10.0]) == pytest.approx(101.0)
    assert execution_vwap([100.0, 110.0], [1.0, 3.0]) == pytest.approx(107.5)


def test_execution_vwap_empty_is_zero() -> None:
    assert execution_vwap([], []) == 0.0


def test_execution_vwap_shape_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="must match"):
        execution_vwap([100.0, 101.0], [10.0])


def test_implementation_shortfall_buy_cost() -> None:
    # paid 101 avg vs 100 arrival on a buy -> 1% cost
    assert implementation_shortfall([100.0, 102.0], [10.0, 10.0], 100.0, "buy") == pytest.approx(
        0.01
    )


def test_implementation_shortfall_sell_cost() -> None:
    # received 99 avg vs 100 arrival on a sell -> 1% cost
    assert implementation_shortfall([98.0, 100.0], [10.0, 10.0], 100.0, "sell") == pytest.approx(
        0.01
    )


def test_implementation_shortfall_price_improvement_is_negative() -> None:
    # bought below arrival -> negative cost (improvement)
    assert implementation_shortfall([99.0], [10.0], 100.0, "buy") == pytest.approx(-0.01)


def test_vwap_slippage_buy_and_sell() -> None:
    assert vwap_slippage([101.0], [10.0], 100.0, "buy") == pytest.approx(0.01)
    assert vwap_slippage([99.0], [10.0], 100.0, "sell") == pytest.approx(0.01)


def test_empty_fills_zero_cost() -> None:
    assert implementation_shortfall([], [], 100.0, "buy") == 0.0
    assert vwap_slippage([], [], 100.0, "buy") == 0.0


def test_invalid_side_raises() -> None:
    with pytest.raises(ValueError, match="side"):
        implementation_shortfall([100.0], [1.0], 100.0, "long")


def test_invalid_benchmarks_raise() -> None:
    with pytest.raises(ValueError, match="arrival_price"):
        implementation_shortfall([100.0], [1.0], 0.0, "buy")
    with pytest.raises(ValueError, match="benchmark_vwap"):
        vwap_slippage([100.0], [1.0], 0.0, "buy")
