"""Scenario stress testing (what-if shocks to a portfolio).

VaR answers "how bad is a typical bad day"; the risk desk's other
standing question is "what does *this specific event* do to the book" —
a 1987-style index gap, a rate shock, a repeat of a historical week.
This module prices a weight vector under explicit shock scenarios:

* :func:`scenario_pnl` — shocks specified per **asset** (fractional
  returns), P&L = ``Σ w_a · shock_a`` per scenario;
* :func:`factor_scenario_pnl` — shocks specified per **factor**,
  propagated through a loadings matrix (``P&L = wᵀ B f``), sharing the
  beta/label conventions of :func:`src.risk.factor_var.factor_model_var`.

Results are per-scenario P&L as a fraction of equity (negative = loss);
``result.idxmin()`` names the worst case. Idiosyncratic moves are
whatever the scenario says — factor scenarios deliberately shock the
systematic part only.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _validated_frame(frame: pd.DataFrame, name: str) -> None:
    if frame.empty:
        raise ValueError(f"{name} must not be empty.")
    if not np.isfinite(frame.to_numpy(dtype=float)).all():
        raise ValueError(f"{name} must be finite.")


def scenario_pnl(weights: pd.Series, scenarios: pd.DataFrame) -> pd.Series:
    """Portfolio P&L under per-asset shock scenarios.

    Args:
        weights: Portfolio weights per asset (fraction-of-equity units,
            shorts negative).
        scenarios: One row per scenario, columns = asset shock returns
            (e.g. -0.10 = that asset falls 10%); must cover every asset
            held.

    Returns:
        Series named ``"pnl"`` indexed by scenario name, in scenario
        order: the fractional equity P&L of each scenario.

    Raises:
        ValueError: If a held asset has no shock column, or any input is
            non-finite/empty.
    """
    _validated_frame(scenarios, "scenarios")
    if not np.isfinite(weights.to_numpy(dtype=float)).all():
        raise ValueError("weights must be finite.")
    missing = [a for a in weights.index if a not in scenarios.columns]
    if missing:
        raise ValueError(f"scenarios missing shock columns for {missing}.")

    shocks = scenarios[list(weights.index)].to_numpy(dtype=float)
    pnl = shocks @ weights.to_numpy(dtype=float)
    return pd.Series(pnl, index=scenarios.index, name="pnl")


def factor_scenario_pnl(
    weights: pd.Series,
    betas: pd.DataFrame,
    factor_scenarios: pd.DataFrame,
) -> pd.Series:
    """Portfolio P&L under factor-level shock scenarios.

    Each factor shock propagates to assets through the loadings
    (``asset shock = B f``), so the scenario P&L is ``wᵀ B f`` — the
    systematic response only, idiosyncratic terms untouched.

    Args:
        weights: Portfolio weights per asset.
        betas: Loadings matrix, rows = assets (must cover the weights),
            columns = factors.
        factor_scenarios: One row per scenario, columns = factor shocks
            (same labels as the beta columns, any order).

    Returns:
        Series named ``"pnl"`` indexed by scenario name.

    Raises:
        ValueError: If assets/factor labels are missing or mismatched, or
            any input is non-finite/empty.
    """
    _validated_frame(factor_scenarios, "factor_scenarios")
    _validated_frame(betas, "betas")
    if not np.isfinite(weights.to_numpy(dtype=float)).all():
        raise ValueError("weights must be finite.")

    missing_assets = [a for a in weights.index if a not in betas.index]
    if missing_assets:
        raise ValueError(f"betas missing assets {missing_assets}.")
    factors = list(betas.columns)
    if set(factor_scenarios.columns) != set(factors):
        raise ValueError("factor_scenarios columns must match the beta columns.")

    loadings = betas.loc[list(weights.index), factors].to_numpy(dtype=float)
    shocks = factor_scenarios[factors].to_numpy(dtype=float)
    exposure = loadings.T @ weights.to_numpy(dtype=float)  # portfolio loading per factor
    pnl = shocks @ exposure
    return pd.Series(pnl, index=factor_scenarios.index, name="pnl")
