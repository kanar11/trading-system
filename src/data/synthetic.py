"""Synthetic OHLCV generation for offline demos and tests.

A geometric Brownian motion (GBM) price path dressed up as a valid OHLCV bar
series: the close follows GBM with annualised drift ``mu`` and volatility
``sigma``; each bar's open gaps from the prior close, and high/low bracket the
open and close with a small intrabar range. The output is always
OHLC-consistent and strictly positive, so it drops straight into the backtest
engine, indicators, or the data-quality auditor without hitting the network.

Pure numpy; deterministic for a given ``seed``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def generate_gbm_ohlcv(
    n_days: int = 252,
    start_price: float = 100.0,
    mu: float = 0.05,
    sigma: float = 0.2,
    seed: int | None = None,
    start: str = "2020-01-01",
    gap_vol: float = 0.002,
    intrabar_vol: float = 0.005,
) -> pd.DataFrame:
    """Generate a synthetic OHLCV frame from a GBM close path.

    Args:
        n_days: Number of business-day bars (>= 1).
        start_price: Price level the path starts from (> 0).
        mu: Annualised drift.
        sigma: Annualised volatility (>= 0; 0 gives a deterministic drift path).
        seed: Seed for reproducibility (None = nondeterministic).
        start: First date (business days are generated forward from here).
        gap_vol: Std of the open-vs-prior-close gap.
        intrabar_vol: Scale of the high/low range around open & close.

    Returns:
        DataFrame indexed by a business-day DatetimeIndex with lowercase
        ``open``/``high``/``low``/``close``/``volume`` columns.

    Raises:
        ValueError: If ``n_days`` < 1, ``start_price`` <= 0, or ``sigma`` < 0.
    """
    if n_days < 1:
        raise ValueError(f"n_days must be >= 1, got {n_days}.")
    if start_price <= 0:
        raise ValueError(f"start_price must be > 0, got {start_price}.")
    if sigma < 0:
        raise ValueError(f"sigma must be >= 0, got {sigma}.")

    rng = np.random.default_rng(seed)
    dt = 1.0 / TRADING_DAYS

    shocks = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * rng.standard_normal(n_days)
    close = start_price * np.exp(np.cumsum(shocks))

    prev_close = np.empty(n_days)
    prev_close[0] = start_price
    prev_close[1:] = close[:-1]
    open_ = prev_close * (1.0 + rng.normal(0.0, gap_vol, n_days))

    hi_base = np.maximum(open_, close)
    lo_base = np.minimum(open_, close)
    up_range = np.clip(np.abs(rng.normal(0.0, intrabar_vol, n_days)), 0.0, 0.5)
    dn_range = np.clip(np.abs(rng.normal(0.0, intrabar_vol, n_days)), 0.0, 0.5)
    high = hi_base * (1.0 + up_range)
    low = lo_base * (1.0 - dn_range)

    volume = rng.integers(1_000_000, 10_000_000, n_days).astype(float)
    index = pd.bdate_range(start=start, periods=n_days)

    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=index,
    )
