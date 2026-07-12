"""Walk-forward validation for strategy robustness testing."""

from src.validation.cpcv import (
    assemble_backtest_paths,
    combinatorial_purged_splits,
    n_backtest_paths,
)
from src.validation.pbo import PBOResult, probability_of_backtest_overfitting
from src.validation.purged_cv import purged_kfold_splits
from src.validation.reality_check import RealityCheckResult, whites_reality_check
from src.validation.timing import TimingTestResult, henriksson_merton, treynor_mazuy
from src.validation.walk_forward import (
    FoldResult,
    WalkForwardConfig,
    print_walk_forward_report,
    run_walk_forward,
)

__all__ = [
    "FoldResult",
    "WalkForwardConfig",
    "print_walk_forward_report",
    "run_walk_forward",
    "PBOResult",
    "probability_of_backtest_overfitting",
    "purged_kfold_splits",
    "combinatorial_purged_splits",
    "n_backtest_paths",
    "assemble_backtest_paths",
    "RealityCheckResult",
    "whites_reality_check",
    "TimingTestResult",
    "treynor_mazuy",
    "henriksson_merton",
]
