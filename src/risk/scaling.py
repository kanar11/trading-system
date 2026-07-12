"""Risk-managed exposure scaling (constant-volatility weighting).

Momentum's fat left tail is concentrated in high-volatility episodes —
the "momentum crashes" of Daniel & Moskowitz (2016). Barroso &
Santa-Clara (2015) show that simply scaling the strategy's exposure by
its own trailing realised volatility, targeting a constant risk level,
roughly doubles momentum's Sharpe ratio and removes most of the crash
risk. The same constant-vol weighting is standard in TSMOM
implementations (Moskowitz, Ooi & Pedersen scale each asset by
40%/σ ex-ante).

This module computes that scaling path for any strategy return series::

    scale_t = min(target_vol / realized_vol_t, max_leverage)

The scalar is *decided* at bar ``t`` close; multiplying the **next**
bar's return follows the package's shift-by-one execution convention
(the caller shifts, exactly like ``signal`` columns). Warm-up bars are
NaN so the caller can distinguish "no estimate yet" from a genuine
scale of 0.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def risk_managed_scaling(
    returns: pd.Series,
    target_vol: float = 0.12,
    window: int = 126,
    max_leverage: float = 2.0,
    periods_per_year: int = 252,
) -> pd.Series:
    """Constant-volatility exposure scalar for a strategy return series.

    Args:
        returns: Per-bar strategy (or asset) returns.
        target_vol: Annualised volatility target (e.g. 0.12 = 12%).
        window: Trailing window for the realised-volatility estimate.
        max_leverage: Cap on the scalar (prevents huge leverage in quiet
            markets).
        periods_per_year: Bars per year for annualisation.

    Returns:
        Series named ``"risk_scale"``: the exposure multiplier decided on
        each bar's close (NaN during the ``window``-bar warm-up; 0 only
        if realised volatility is infinite).

    Raises:
        ValueError: If ``target_vol``/``max_leverage`` <= 0, ``window`` < 2
            or ``periods_per_year`` < 1.
    """
    if target_vol <= 0:
        raise ValueError(f"target_vol must be > 0, got {target_vol}.")
    if max_leverage <= 0:
        raise ValueError(f"max_leverage must be > 0, got {max_leverage}.")
    if window < 2:
        raise ValueError(f"window must be >= 2, got {window}.")
    if periods_per_year < 1:
        raise ValueError(f"periods_per_year must be >= 1, got {periods_per_year}.")

    realized = returns.rolling(window).std(ddof=1) * float(np.sqrt(periods_per_year))
    scale = (target_vol / realized).clip(upper=max_leverage)
    return scale.rename("risk_scale")


def apply_risk_scaling(
    returns: pd.Series,
    scale: pd.Series,
) -> pd.Series:
    """Apply an exposure scalar to a return series with next-bar execution.

    Each bar's return is multiplied by the scalar decided on the *previous*
    bar's close (``scale.shift(1)``); bars with no prior estimate (warm-up)
    keep a scale of 0, i.e. no exposure before the model is ready.

    Args:
        returns: Per-bar strategy returns.
        scale: Scalar path from :func:`risk_managed_scaling`, on the same
            index.

    Returns:
        The scaled return series (same index/name as ``returns``).

    Raises:
        ValueError: If the indexes differ.
    """
    if not returns.index.equals(scale.index):
        raise ValueError("returns and scale must share the same index.")
    return returns * scale.shift(1).fillna(0.0)
