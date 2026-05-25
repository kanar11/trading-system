"""Tests for the Strategy base class and SmaCrossoverStrategy."""

import numpy as np
import pandas as pd
import pytest

from src.backtest.event_engine import EventEngine
from src.strategy.base import Strategy, SmaCrossoverStrategy


def _trending_ohlc(n: int = 200, slope: float = 0.5, seed: int = 0) -> pd.DataFrame:
    """Synthetic OHLCV with a strong upward drift — SMA crossover should profit."""
    rng = np.random.default_rng(seed)
    noise = rng.normal(0, 0.5, n)
    closes = 100 + np.arange(n) * slope + noise
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    return pd.DataFrame(
        {
            "open": closes,
            "high": closes * 1.005,
            "low": closes * 0.995,
            "close": closes,
        },
        index=dates,
    )


class TestStrategyABC:
    def test_cannot_instantiate_abstract_strategy(self):
        with pytest.raises(TypeError):
            Strategy()  # has @abstractmethod on_bar


class TestSmaCrossover:
    def test_validates_fast_lt_slow(self):
        with pytest.raises(ValueError, match="fast"):
            SmaCrossoverStrategy(fast=30, slow=10)

    def test_runs_end_to_end_on_event_engine(self):
        df = _trending_ohlc(n=150, slope=0.3)
        eng = EventEngine(symbol="ASSET", initial_cash=100_000)
        strat = SmaCrossoverStrategy(fast=5, slow=20, trade_qty=10)
        res = eng.run(df, strat)
        # equity curve length matches bars
        assert len(res.equity_curve) == 150
        # at least one fill happened in 150 bars of trending data
        assert not res.fills.empty
        # final equity differs from initial (something traded)
        assert res.equity_curve.iloc[-1] != 100_000

    def test_does_not_trade_during_warmup(self):
        df = _trending_ohlc(n=10)  # fewer bars than slow window
        eng = EventEngine(symbol="ASSET", initial_cash=100_000)
        strat = SmaCrossoverStrategy(fast=3, slow=30)
        res = eng.run(df, strat)
        assert res.fills.empty

    def test_short_disabled_clamps_to_flat(self):
        # falling series — would normally trigger short signals
        rng = np.random.default_rng(0)
        n = 100
        closes = 200 - np.arange(n) * 0.5 + rng.normal(0, 0.2, n)
        dates = pd.date_range("2020-01-01", periods=n, freq="B")
        df = pd.DataFrame(
            {"open": closes, "high": closes * 1.01, "low": closes * 0.99, "close": closes},
            index=dates,
        )
        eng = EventEngine(symbol="ASSET", initial_cash=100_000)
        strat = SmaCrossoverStrategy(fast=5, slow=20, allow_short=False)
        res = eng.run(df, strat)
        # the position should never go short
        pos = res.portfolio.positions.get("ASSET")
        if pos is not None:
            assert pos.quantity >= 0

    def test_lifecycle_hooks_default_to_noop(self):
        df = _trending_ohlc(n=60)
        eng = EventEngine(symbol="ASSET", initial_cash=100_000)

        class Counter(Strategy):
            calls = 0
            def on_bar(self, ctx): self.calls += 1

        strat = Counter()
        # default on_start / on_end / on_order_event don't blow up
        strat.on_start.__call__  # exists
        eng.run(df, strat)
        assert strat.calls == 60
