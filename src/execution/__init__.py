"""Realistic execution-cost modelling (spread + market impact)."""

from src.execution.slippage import (
    ExecutionConfig,
    compute_execution_cost,
    apply_execution_costs,
)

__all__ = ["ExecutionConfig", "compute_execution_cost", "apply_execution_costs"]
