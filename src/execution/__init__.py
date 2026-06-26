"""Realistic execution-cost modelling (spread + market impact)."""

from src.execution.slippage import (
    ExecutionConfig,
    apply_execution_costs,
    compute_execution_cost,
)

__all__ = ["ExecutionConfig", "compute_execution_cost", "apply_execution_costs"]
