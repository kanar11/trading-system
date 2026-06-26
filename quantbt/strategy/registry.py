"""Strategy registry: map a strategy name to a signal-generating function.

Strategies register themselves with the :func:`register` decorator, so adding a
new strategy is a matter of writing a builder and decorating it — no changes to
the CLI or dispatch logic. :func:`build_strategy` turns a validated
:class:`~quantbt.config.RunConfig` into a ready-to-run ``df -> df`` callable.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import partial

import pandas as pd

from quantbt.config import RunConfig
from quantbt.regime.detector import adaptive_strategy
from quantbt.strategy.mean_reversion import mean_reversion_strategy
from quantbt.strategy.momentum import momentum_strategy

StrategyFn = Callable[[pd.DataFrame], pd.DataFrame]
StrategyBuilder = Callable[[RunConfig], StrategyFn]

_REGISTRY: dict[str, StrategyBuilder] = {}


def register(name: str) -> Callable[[StrategyBuilder], StrategyBuilder]:
    """Register a strategy builder under ``name`` (raises on duplicates)."""

    def decorator(builder: StrategyBuilder) -> StrategyBuilder:
        if name in _REGISTRY:
            raise ValueError(f"Strategy '{name}' is already registered.")
        _REGISTRY[name] = builder
        return builder

    return decorator


def available() -> list[str]:
    """Return the sorted list of registered strategy names."""
    return sorted(_REGISTRY)


def build_strategy(config: RunConfig) -> StrategyFn:
    """Build the signal-generating function for ``config.strategy``."""
    try:
        builder = _REGISTRY[config.strategy]
    except KeyError:
        raise ValueError(
            f"Unknown strategy '{config.strategy}'. Available: {available()}."
        ) from None
    return builder(config)


@register("momentum")
def _build_momentum(config: RunConfig) -> StrategyFn:
    p = config.momentum
    return partial(
        momentum_strategy,
        lookback=p.lookback,
        threshold=p.threshold,
        use_sma_filter=p.use_sma_filter,
    )


@register("mean-reversion")
def _build_mean_reversion(config: RunConfig) -> StrategyFn:
    p = config.mean_reversion
    return partial(
        mean_reversion_strategy,
        bb_window=p.bb_window,
        bb_std=p.bb_std,
        rsi_period=p.rsi_period,
        rsi_oversold=p.rsi_oversold,
        rsi_overbought=p.rsi_overbought,
        use_rsi_filter=p.use_rsi_filter,
    )


@register("adaptive")
def _build_adaptive(config: RunConfig) -> StrategyFn:
    return partial(
        adaptive_strategy,
        momentum_fn=_build_momentum(config),
        mean_reversion_fn=_build_mean_reversion(config),
        config=config.regime.to_dataclass(),
    )
