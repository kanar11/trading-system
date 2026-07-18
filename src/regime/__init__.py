"""Market regime detection and adaptive strategy selection."""

from src.regime.conditional import regime_performance
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
from src.regime.hmm_filter import filter_hmm_probabilities, filtered_hmm_states
from src.regime.market_states import BEAR, BULL, bull_bear_labels
from src.regime.transitions import (
    forecast_regime_probabilities,
    markov_entropy_rate,
    regime_durations,
    regime_predictability,
    regime_transition_matrix,
    stationary_distribution,
)
from src.regime.turbulence import financial_turbulence, turbulent_periods
from src.regime.volatility import VolRegime, realized_volatility, vol_regimes

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
    "stationary_distribution",
    "forecast_regime_probabilities",
    "markov_entropy_rate",
    "regime_predictability",
    "financial_turbulence",
    "turbulent_periods",
    "VolRegime",
    "realized_volatility",
    "vol_regimes",
    "regime_performance",
    "BULL",
    "BEAR",
    "bull_bear_labels",
    "filter_hmm_probabilities",
    "filtered_hmm_states",
]
