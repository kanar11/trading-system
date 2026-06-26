"""Tests for the strategy registry."""

import pandas as pd
import pytest

from quantbt.config import RunConfig
from quantbt.strategy import available, build_strategy, register
from quantbt.strategy.registry import _REGISTRY, StrategyFn


def test_available_lists_builtin_strategies() -> None:
    assert available() == ["adaptive", "mean-reversion", "momentum"]


@pytest.mark.parametrize("name", ["momentum", "mean-reversion", "adaptive"])
def test_build_strategy_produces_signals(name: str, sample_ohlcv: pd.DataFrame) -> None:
    cfg = RunConfig.model_validate({"strategy": name})
    strategy_fn = build_strategy(cfg)
    out = strategy_fn(sample_ohlcv)
    assert "signal" in out.columns
    assert set(out["signal"].unique()).issubset({-1, 0, 1})


def test_build_strategy_respects_params(sample_ohlcv: pd.DataFrame) -> None:
    cfg = RunConfig.model_validate(
        {"strategy": "momentum", "momentum": {"lookback": 5, "threshold": 0.0}}
    )
    out = build_strategy(cfg)(sample_ohlcv)
    assert "signal" in out.columns


def test_register_duplicate_raises() -> None:
    with pytest.raises(ValueError, match="already registered"):

        @register("momentum")
        def _dupe(config: RunConfig) -> StrategyFn:  # pragma: no cover - body never runs
            raise NotImplementedError


def test_register_adds_to_available() -> None:
    @register("noop")
    def _noop(config: RunConfig) -> StrategyFn:  # pragma: no cover - builder not invoked
        return lambda df: df.assign(signal=0)

    try:
        assert "noop" in available()
        assert _REGISTRY["noop"] is _noop
    finally:
        _REGISTRY.pop("noop", None)
    assert "noop" not in available()
