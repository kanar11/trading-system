"""Realistic execution-cost modelling (spread + market impact)."""

from src.execution.financing import apply_financing, financing_costs
from src.execution.impact import (
    almgren_chriss_cost,
    almgren_chriss_trajectory,
    participation_rate_cost,
)
from src.execution.slippage import (
    ExecutionConfig,
    apply_execution_costs,
    compute_execution_cost,
)
from src.execution.tca import (
    execution_vwap,
    implementation_shortfall,
    vwap_slippage,
)

__all__ = [
    "ExecutionConfig",
    "compute_execution_cost",
    "apply_execution_costs",
    "participation_rate_cost",
    "almgren_chriss_trajectory",
    "almgren_chriss_cost",
    "execution_vwap",
    "implementation_shortfall",
    "vwap_slippage",
    "financing_costs",
    "apply_financing",
]
