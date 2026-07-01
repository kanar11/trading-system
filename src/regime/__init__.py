"""Market regime detection and adaptive strategy selection."""

from src.regime.detector import (
    RegimeConfig,
    RegimeType,
    adaptive_strategy,
    detect_regime,
)
from src.regime.hmm import (
    HMMConfig,
    HMMResult,
    detect_hmm_regime,
    fit_gaussian_hmm,
)
from src.regime.transitions import regime_durations, regime_transition_matrix

__all__ = [
    "RegimeConfig",
    "RegimeType",
    "adaptive_strategy",
    "detect_regime",
    "HMMConfig",
    "HMMResult",
    "detect_hmm_regime",
    "fit_gaussian_hmm",
    "regime_transition_matrix",
    "regime_durations",
]
