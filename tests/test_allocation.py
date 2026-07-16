"""Tests for pro-rata block-trade allocation."""

import numpy as np
import pytest

from src.oms import pro_rata_allocation


def test_exact_division_matches_weights() -> None:
    out = pro_rata_allocation(10, {"a": 0.5, "b": 0.3, "c": 0.2})
    assert out == {"a": 5.0, "b": 3.0, "c": 2.0}


def test_largest_remainder_gets_the_leftover_lot() -> None:
    # exact shares: a=4.4, b=3.3, c=3.3 -> a rounds down to 4, leftover lot
    # goes to the largest remainder (a: 0.4 > 0.3)
    out = pro_rata_allocation(11, {"a": 0.4, "b": 0.3, "c": 0.3})
    assert out["a"] == 5.0
    assert sum(out.values()) == 11.0


def test_ties_break_by_account_name() -> None:
    # equal weights, 11 lots: shares 3.667 each -> two leftover lots go to
    # the alphabetically first names among equal remainders
    out = pro_rata_allocation(11, {"c": 1.0, "a": 1.0, "b": 1.0})
    assert out == {"a": 4.0, "b": 4.0, "c": 3.0}


def test_zero_weight_account_gets_nothing() -> None:
    out = pro_rata_allocation(10, {"live": 1.0, "closed": 0.0})
    assert out["closed"] == 0.0
    assert out["live"] == 10.0


def test_board_lots_and_sub_lot_residue() -> None:
    out = pro_rata_allocation(1_050, {"a": 0.5, "b": 0.5}, lot_size=100.0)
    assert sum(out.values()) == 1_000.0  # 50-share residue stays unallocated
    assert set(out.values()) == {500.0}
    assert all(q % 100.0 == 0 for q in out.values())


def test_sum_property_on_random_weights() -> None:
    rng = np.random.default_rng(2)
    for _ in range(25):
        weights = {f"acct{i}": float(w) for i, w in enumerate(rng.random(6))}
        total = float(rng.integers(1, 500))
        out = pro_rata_allocation(total, weights)
        assert sum(out.values()) == pytest.approx(np.floor(total))
        # nobody drifts more than one lot from the exact share
        exact = {a: w / sum(weights.values()) * np.floor(total) for a, w in weights.items()}
        assert all(abs(out[a] - exact[a]) < 1.0 + 1e-9 for a in weights)


def test_zero_quantity_allocates_zeros() -> None:
    out = pro_rata_allocation(0.0, {"a": 1.0, "b": 1.0})
    assert out == {"a": 0.0, "b": 0.0}


def test_bad_inputs_raise() -> None:
    with pytest.raises(ValueError, match="total_quantity"):
        pro_rata_allocation(-1, {"a": 1.0})
    with pytest.raises(ValueError, match="lot_size"):
        pro_rata_allocation(10, {"a": 1.0}, lot_size=0.0)
    with pytest.raises(ValueError, match="empty"):
        pro_rata_allocation(10, {})
    with pytest.raises(ValueError, match="finite"):
        pro_rata_allocation(10, {"a": float("nan")})
    with pytest.raises(ValueError, match="at least one"):
        pro_rata_allocation(10, {"a": 0.0, "b": 0.0})
