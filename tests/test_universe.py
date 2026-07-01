"""Tests for universe helpers."""

import pytest

from src.data.universe import (
    FAANG,
    FAANG_PLUS,
    combine_universes,
    get_universe,
    in_universe,
    list_universes,
)


def test_list_universes() -> None:
    assert list_universes() == [
        "benchmarks",
        "dow30",
        "faang",
        "faang_plus",
        "factors",
        "sectors",
    ]


def test_get_universe_defensive_copy() -> None:
    u = get_universe("faang")
    u.append("XXX")
    assert "XXX" not in get_universe("faang")  # original not mutated


def test_combine_dedupes_first_occurrence() -> None:
    # FAANG is a subset of FAANG_PLUS -> union equals FAANG_PLUS order
    assert combine_universes("faang", "faang_plus") == FAANG_PLUS


def test_combine_without_dedupe_keeps_all() -> None:
    combined = combine_universes("faang", "faang_plus", dedupe=False)
    assert len(combined) == len(FAANG) + len(FAANG_PLUS)


def test_combine_result_is_unique_by_default() -> None:
    combined = combine_universes("dow30", "faang", "benchmarks")
    assert len(combined) == len(set(combined))


def test_in_universe_case_insensitive() -> None:
    assert in_universe("AAPL", "faang")
    assert in_universe("aapl", "faang")
    assert not in_universe("TSLA", "faang")
    assert in_universe("TSLA", "faang_plus")


def test_unknown_universe_raises() -> None:
    with pytest.raises(KeyError, match="Unknown universe"):
        combine_universes("faang", "nope")
    with pytest.raises(KeyError, match="Unknown universe"):
        in_universe("AAPL", "nope")
