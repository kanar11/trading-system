"""Regime-conditional performance statistics.

Detecting regimes is only half the job — the payoff is knowing how a
strategy (or the market) behaves *inside* each one. This module produces
the standard regime-attribution table: per regime label, how much of the
sample it covers and what the returns looked like while it was active.

Works with any labelling on the same index as the returns — the HMM states
of :func:`~src.regime.hmm.detect_hmm_regime`, the LOW/NORMAL/HIGH codes of
:func:`~src.regime.volatility.vol_regimes`, the boolean flags of
:func:`~src.regime.turbulence.turbulent_periods`, or hand-made labels.
Bars whose label is NaN (detector warm-up) are excluded and reported via
the ``share`` column summing to less than 1.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def regime_performance(
    returns: pd.Series,
    regimes: pd.Series,
    periods_per_year: int = 252,
) -> pd.DataFrame:
    """Per-regime return statistics over an aligned label series.

    Args:
        returns: Per-bar returns.
        regimes: Regime label per bar (int, str or bool), on exactly the
            same index as ``returns``; NaN labels are skipped.
        periods_per_year: Bars per year for annualisation.

    Returns:
        DataFrame indexed by regime label (sorted) with columns:

        * ``n_obs`` — bars spent in the regime.
        * ``share`` — fraction of all bars (NaN-labelled bars excluded, so
          the column sums to <= 1).
        * ``ann_return`` — mean bar return × ``periods_per_year``.
        * ``ann_vol`` — bar-return std (ddof=1) × sqrt(``periods_per_year``);
          NaN for single-bar regimes.
        * ``sharpe`` — ``ann_return / ann_vol``; NaN when the volatility is
          zero or undefined.
        * ``hit_rate`` — fraction of positive bars.
        * ``best`` / ``worst`` — extreme single-bar returns.

    Raises:
        ValueError: If the series are empty, the indexes differ, or
            ``periods_per_year`` < 1.
    """
    if len(returns) == 0:
        raise ValueError("returns must not be empty.")
    if not returns.index.equals(regimes.index):
        raise ValueError("returns and regimes must share the same index.")
    if periods_per_year < 1:
        raise ValueError(f"periods_per_year must be >= 1, got {periods_per_year}.")

    valid = regimes.notna()
    values = returns[valid]
    labels = regimes[valid]
    if len(values) == 0:
        raise ValueError("regimes contains no non-NaN labels.")

    n_total = len(returns)
    rows = []
    for label in sorted(labels.unique()):
        in_regime = values[labels == label]
        mean = float(in_regime.mean())
        std = float(in_regime.std(ddof=1)) if len(in_regime) > 1 else float("nan")
        ann_return = mean * periods_per_year
        ann_vol = std * float(np.sqrt(periods_per_year))
        sharpe = ann_return / ann_vol if np.isfinite(ann_vol) and ann_vol > 0 else float("nan")
        rows.append(
            {
                "regime": label,
                "n_obs": len(in_regime),
                "share": len(in_regime) / n_total,
                "ann_return": ann_return,
                "ann_vol": ann_vol,
                "sharpe": sharpe,
                "hit_rate": float((in_regime > 0).mean()),
                "best": float(in_regime.max()),
                "worst": float(in_regime.min()),
            }
        )
    return pd.DataFrame(rows).set_index("regime")
