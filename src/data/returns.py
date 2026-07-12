"""Return-series transforms: simple, log, excess, and back to prices.

Every research pipeline keeps re-deriving the same four transforms — and
subtly disagreeing about them (arithmetic vs geometric de-annualisation of
the risk-free rate, log vs simple compounding). This module pins the
conventions down once, for both Series and wide DataFrames:

* ``simple_returns`` / ``log_returns`` — bar-over-bar changes (first bar
  NaN); log returns require strictly positive prices.
* ``simple_to_log`` / ``log_to_simple`` — exact inverses via
  ``log1p`` / ``expm1``.
* ``returns_to_prices`` — compound simple returns back into a price/equity
  path (round-trips ``simple_returns`` given the initial price).
* ``excess_returns`` — subtract a risk-free rate given either as an
  *annual* scalar (de-annualised geometrically) or a per-bar series; the
  input momentum filters like GEM (#47) actually want.

Direct-import module::

    from src.data.returns import simple_returns, excess_returns
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _wrap(values: np.ndarray, template: pd.Series | pd.DataFrame) -> pd.Series | pd.DataFrame:
    """Rebuild a Series/DataFrame around ``values`` using ``template`` labels."""
    if isinstance(template, pd.DataFrame):
        return pd.DataFrame(values, index=template.index, columns=template.columns)
    return pd.Series(values, index=template.index, name=template.name)


def _validated_prices(prices: pd.Series | pd.DataFrame) -> np.ndarray:
    arr = prices.to_numpy(dtype=float)
    if np.isnan(arr).any() or (arr <= 0).any():
        raise ValueError("prices must be positive and NaN-free.")
    return arr


def simple_returns(prices: pd.Series | pd.DataFrame) -> pd.Series | pd.DataFrame:
    """Bar-over-bar simple returns ``p_t / p_{t-1} - 1`` (first bar NaN).

    Raises:
        ValueError: If prices are non-positive or NaN.
    """
    arr = _validated_prices(prices)
    out = np.full_like(arr, np.nan)
    out[1:] = arr[1:] / arr[:-1] - 1.0
    return _wrap(out, prices)


def log_returns(prices: pd.Series | pd.DataFrame) -> pd.Series | pd.DataFrame:
    """Bar-over-bar log returns ``ln(p_t / p_{t-1})`` (first bar NaN).

    Raises:
        ValueError: If prices are non-positive or NaN.
    """
    arr = _validated_prices(prices)
    out = np.full_like(arr, np.nan)
    out[1:] = np.log(arr[1:] / arr[:-1])
    return _wrap(out, prices)


def simple_to_log(returns: pd.Series | pd.DataFrame) -> pd.Series | pd.DataFrame:
    """Convert simple returns to log returns (``log1p``); NaNs pass through."""
    return _wrap(np.log1p(returns.to_numpy(dtype=float)), returns)


def log_to_simple(returns: pd.Series | pd.DataFrame) -> pd.Series | pd.DataFrame:
    """Convert log returns to simple returns (``expm1``); NaNs pass through."""
    return _wrap(np.expm1(returns.to_numpy(dtype=float)), returns)


def returns_to_prices(
    returns: pd.Series | pd.DataFrame,
    initial: float = 1.0,
) -> pd.Series | pd.DataFrame:
    """Compound simple returns into a price/equity path.

    ``returns_to_prices(simple_returns(p).dropna(), initial=p[0])`` recovers
    ``p`` from its second bar on.

    Args:
        returns: NaN-free simple returns (drop the warm-up first).
        initial: Starting price/equity level (> 0).

    Raises:
        ValueError: If ``initial`` <= 0 or the returns contain NaNs.
    """
    if initial <= 0:
        raise ValueError(f"initial must be > 0, got {initial}.")
    arr = returns.to_numpy(dtype=float)
    if np.isnan(arr).any():
        raise ValueError("returns must be NaN-free (drop the warm-up bar first).")
    return _wrap(initial * np.cumprod(1.0 + arr, axis=0), returns)


def excess_returns(
    returns: pd.Series | pd.DataFrame,
    risk_free: float | pd.Series = 0.0,
    periods_per_year: int = 252,
) -> pd.Series | pd.DataFrame:
    """Subtract the risk-free rate from a return series.

    Args:
        returns: Per-bar simple returns (Series or wide DataFrame).
        risk_free: Either an *annual* rate (de-annualised geometrically to
            ``(1 + rf)^(1/periods_per_year) - 1`` per bar) or a per-bar
            rate Series on the same index.
        periods_per_year: Bars per year for the scalar conversion.

    Returns:
        Excess returns with the input's labels.

    Raises:
        ValueError: If ``periods_per_year`` < 1, a scalar rate is <= -1,
            or a rate series is on a different index.
    """
    if periods_per_year < 1:
        raise ValueError(f"periods_per_year must be >= 1, got {periods_per_year}.")

    if isinstance(risk_free, pd.Series):
        if not returns.index.equals(risk_free.index):
            raise ValueError("returns and risk_free must share the same index.")
        per_bar = risk_free.to_numpy(dtype=float)
    else:
        if risk_free <= -1.0:
            raise ValueError(f"risk_free must be > -1, got {risk_free}.")
        per_bar = np.full(len(returns), float(1.0 + risk_free) ** (1.0 / periods_per_year) - 1.0)

    arr = returns.to_numpy(dtype=float)
    if arr.ndim == 2:
        return _wrap(arr - per_bar[:, np.newaxis], returns)
    return _wrap(arr - per_bar, returns)
