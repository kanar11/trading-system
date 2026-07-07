"""Financing costs: short borrow fees and margin interest.

Execution costs (spread, impact) are charged when a position *changes*;
financing costs accrue for as long as it is *held*. Two flows matter for a
long/short book: the stock-loan **borrow fee** paid on short notional, and
**margin interest** paid on the notional financed above the account's own
equity (gross leverage beyond 1x). Both are quoted as annualised rates and
accrued per bar, expressed — like everything in the vectorised engine — as a
return drag in fraction-of-equity units.

Positions can be a single signed exposure Series (the engine's
``scaled_position``) or a wide weight DataFrame (one column per asset, as
produced by :func:`src.strategy.dual_momentum.dual_momentum_strategy`); for
frames, shorts and gross leverage are summed across assets per bar.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def financing_costs(
    positions: pd.Series | pd.DataFrame,
    borrow_rate: float = 0.0,
    margin_rate: float = 0.0,
    periods_per_year: int = 252,
) -> pd.Series:
    """Per-bar financing drag for a held exposure path.

    Per bar::

        short_notional = sum of |negative exposures|
        financed       = max(gross_exposure - 1, 0)
        cost           = short_notional * borrow_rate / periods_per_year
                       + financed * margin_rate / periods_per_year

    An unlevered long-only book (gross <= 1, no shorts) costs nothing.

    Args:
        positions: Signed exposure in fraction-of-equity units — a Series
            (single instrument) or a wide DataFrame (one column per asset).
        borrow_rate: Annualised stock-loan fee charged on short notional.
        margin_rate: Annualised interest on gross exposure above 1x.
        periods_per_year: Bars per year for the per-bar accrual.

    Returns:
        Series of per-bar cost fractions (>= 0), named ``"financing_cost"``.

    Raises:
        ValueError: If a rate is negative, ``periods_per_year`` < 1, or
            ``positions`` contains NaNs.
    """
    if borrow_rate < 0:
        raise ValueError(f"borrow_rate must be >= 0, got {borrow_rate}.")
    if margin_rate < 0:
        raise ValueError(f"margin_rate must be >= 0, got {margin_rate}.")
    if periods_per_year < 1:
        raise ValueError(f"periods_per_year must be >= 1, got {periods_per_year}.")

    values = positions.to_numpy(dtype=float)
    if np.isnan(values).any():
        raise ValueError("positions must not contain NaNs.")
    if values.ndim == 1:
        values = values[:, np.newaxis]

    short_notional = np.clip(-values, 0.0, None).sum(axis=1)
    gross = np.abs(values).sum(axis=1)
    financed = np.clip(gross - 1.0, 0.0, None)

    per_bar = (short_notional * borrow_rate + financed * margin_rate) / periods_per_year
    return pd.Series(per_bar, index=positions.index, name="financing_cost")


def apply_financing(
    returns: pd.Series,
    positions: pd.Series | pd.DataFrame,
    borrow_rate: float = 0.0,
    margin_rate: float = 0.0,
    periods_per_year: int = 252,
) -> pd.Series:
    """Subtract per-bar financing costs from a strategy return series.

    Args:
        returns: Per-bar strategy returns (fraction of equity).
        positions: Exposure path held over those bars — same index as
            ``returns`` (see :func:`financing_costs`).
        borrow_rate: Annualised stock-loan fee on short notional.
        margin_rate: Annualised interest on gross exposure above 1x.
        periods_per_year: Bars per year for the per-bar accrual.

    Returns:
        Net return Series (same index/name as ``returns``).

    Raises:
        ValueError: If the indexes differ or the inputs are invalid.
    """
    if not returns.index.equals(positions.index):
        raise ValueError("returns and positions must share the same index.")
    costs = financing_costs(
        positions,
        borrow_rate=borrow_rate,
        margin_rate=margin_rate,
        periods_per_year=periods_per_year,
    )
    return returns - costs
