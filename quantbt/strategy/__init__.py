"""Trading signal generators and the strategy registry."""

from quantbt.strategy.mean_reversion import mean_reversion_strategy
from quantbt.strategy.momentum import momentum_strategy

# Imported last: the registry pulls in the strategy functions above.
from quantbt.strategy.registry import (  # noqa: E402
    available,
    build_strategy,
    register,
)

__all__ = [
    "momentum_strategy",
    "mean_reversion_strategy",
    "build_strategy",
    "register",
    "available",
]
