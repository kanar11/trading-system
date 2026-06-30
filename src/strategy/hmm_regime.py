"""HMM regime-switching strategy.

Fits a Gaussian hidden Markov model (:mod:`src.regime.hmm`) to the return
series and trades the latent regime directly: long in the highest-mean state
(the "risk-on" regime), short in the lowest-mean state, flat in any middle
state. States are relabelled by ascending mean, so the mapping is stable.

Caveat — the HMM is fit on the *whole* series (Viterbi decoding is in-sample),
so the regime labels peek at the full history. Use it inside walk-forward
validation (which refits per fold) for leakage-controlled out-of-sample results,
or as a research lens on historical regimes.
"""

from __future__ import annotations

import logging

import pandas as pd

from src.regime import detect_hmm_regime

logger = logging.getLogger(__name__)


def hmm_regime_strategy(
    df: pd.DataFrame,
    n_states: int = 2,
    allow_short: bool = True,
) -> pd.DataFrame:
    """Trade Gaussian-HMM return regimes.

    Signal logic (states sorted by ascending mean):
        +1 (long)  in the highest-mean state
        -1 (short) in the lowest-mean state (if ``allow_short``)
         0 (flat)  in any middle state or during the warm-up

    Args:
        df: DataFrame with a 'close' column.
        n_states: Number of HMM regimes (>= 2).
        allow_short: If False, the low-mean regime is flat instead of short.

    Returns:
        DataFrame with ``hmm_state`` and ``signal`` columns.

    Raises:
        ValueError: If 'close' is missing or ``n_states`` < 2.
    """
    if "close" not in df.columns:
        raise ValueError("DataFrame must contain a 'close' column.")
    if n_states < 2:
        raise ValueError(f"n_states must be >= 2, got {n_states}.")

    df = df.copy()
    returns = df["close"].pct_change()
    states = detect_hmm_regime(returns, n_states=n_states)
    df["hmm_state"] = states.reindex(df.index)

    high_state = n_states - 1
    signal = pd.Series(0, index=states.index, dtype=int)
    signal[states == high_state] = 1
    if allow_short:
        signal[states == 0] = -1

    df["signal"] = signal.reindex(df.index, fill_value=0).astype(int)
    return df
