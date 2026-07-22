"""Volatility-regime classification with hysteresis.

Splits a return series into LOW / NORMAL / HIGH volatility regimes by
comparing realised volatility to its own *trailing* distribution — the
thresholds are rolling quantiles, so the classification is causal (no
look-ahead) and adapts as the volatility level of the market drifts over
the years.

A naive quantile cut flips regime on every wobble around the threshold.
The classifier here is a small state machine with **hysteresis**: entering
HIGH requires volatility above the high quantile, but leaving it requires
falling back below the rolling *median* (and symmetrically for LOW). Bars
in between keep the previous regime, which produces the persistent,
low-churn regime labels that sizing and strategy switching actually need.

Complements the other detectors in this package: :mod:`~src.regime.hmm`
(unsupervised, in-sample), :mod:`~src.regime.turbulence` (cross-asset
dislocation) and :mod:`~src.regime.detector` (trend/ADX) — this one is the
plain, causal single-series volatility view.
"""

from __future__ import annotations

from enum import IntEnum

import numpy as np
import pandas as pd


class VolRegime(IntEnum):
    """Volatility regime codes (ordered: LOW < NORMAL < HIGH)."""

    LOW = 0
    NORMAL = 1
    HIGH = 2


def realized_volatility(
    returns: pd.Series,
    window: int = 20,
    periods_per_year: int = 252,
) -> pd.Series:
    """Annualised rolling realised volatility of a return series.

    Args:
        returns: Per-bar returns.
        window: Rolling standard-deviation window (>= 2).
        periods_per_year: Bars per year for annualisation.

    Returns:
        Series named ``"realized_vol"`` (NaN during the warm-up).

    Raises:
        ValueError: If ``window`` < 2 or ``periods_per_year`` < 1.
    """
    if window < 2:
        raise ValueError(f"window must be >= 2, got {window}.")
    if periods_per_year < 1:
        raise ValueError(f"periods_per_year must be >= 1, got {periods_per_year}.")
    vol = returns.rolling(window).std() * float(np.sqrt(periods_per_year))
    return vol.rename("realized_vol")


def vol_of_vol(
    returns: pd.Series,
    vol_window: int = 20,
    vov_window: int = 20,
    periods_per_year: int = 252,
    relative: bool = False,
) -> pd.Series:
    """Volatility of volatility — how *unstable* the volatility regime is.

    A market can be quietly volatile (a high but steady vol) or
    treacherously volatile (a vol that itself lurches). Vol-of-vol
    captures the second: the rolling standard deviation of the annualised
    realised-volatility path (:func:`realized_volatility`). It spikes
    around regime transitions — precisely when a constant-exposure book is
    most likely to be caught wrong-footed — and sits near zero when the
    volatility level is steady, high or low.

    Args:
        returns: Per-bar returns.
        vol_window: Window of the inner realised-volatility estimate (>= 2).
        vov_window: Window of the outer std-of-vol (>= 2).
        periods_per_year: Bars per year for annualising the inner vol.
        relative: If True, divide by the trailing mean vol to get a
            unit-free coefficient of variation of volatility (comparable
            across assets of different vol levels).

    Returns:
        Series named ``"vol_of_vol"`` (NaN during the warm-up).

    Raises:
        ValueError: If either window is < 2 or ``periods_per_year`` < 1.
    """
    if vov_window < 2:
        raise ValueError(f"vov_window must be >= 2, got {vov_window}.")

    vol = realized_volatility(returns, window=vol_window, periods_per_year=periods_per_year)
    vov = vol.rolling(vov_window).std()
    if relative:
        mean_vol = vol.rolling(vov_window).mean().replace(0.0, np.nan)
        vov = vov / mean_vol
    return vov.rename("vol_of_vol")


def _run_state_machine(
    vol: np.ndarray,
    q_low: np.ndarray,
    q_mid: np.ndarray,
    q_high: np.ndarray,
) -> np.ndarray:
    """Hysteresis classification of a volatility path against thresholds.

    Enter HIGH above ``q_high``; leave it only below ``q_mid``. Enter LOW
    below ``q_low``; leave it only above ``q_mid``. Bars with NaN inputs
    keep the previous state (NORMAL at the start).
    """
    n = len(vol)
    states = np.full(n, int(VolRegime.NORMAL))
    state = VolRegime.NORMAL
    for i in range(n):
        v, lo, mid, hi = vol[i], q_low[i], q_mid[i], q_high[i]
        if np.isnan(v) or np.isnan(lo) or np.isnan(mid) or np.isnan(hi):
            states[i] = int(state)
            continue
        if (state is VolRegime.HIGH and v < mid) or (state is VolRegime.LOW and v > mid):
            state = VolRegime.NORMAL
        if state is VolRegime.NORMAL:
            if v > hi:
                state = VolRegime.HIGH
            elif v < lo:
                state = VolRegime.LOW
        states[i] = int(state)
    return states


def vol_regimes(
    returns: pd.Series,
    window: int = 20,
    lookback: int = 252,
    low_quantile: float = 0.10,
    high_quantile: float = 0.90,
    periods_per_year: int = 252,
) -> pd.Series:
    """Classify each bar into a LOW / NORMAL / HIGH volatility regime.

    Realised volatility (rolling ``window`` std, annualised) is compared to
    its own trailing ``lookback``-bar quantiles; the hysteresis state
    machine (see :func:`_run_state_machine`) turns the comparison into
    persistent regime labels. Warm-up bars are NORMAL.

    Because the thresholds are *relative*, the classifier flags the calmest
    and wildest spells of recent memory even in a statistically stationary
    market — expect some time in the extreme regimes at all times, with the
    labels aligning strongly with genuine volatility shifts.

    Args:
        returns: Per-bar returns.
        window: Realised-volatility window (>= 2).
        lookback: Trailing window for the quantile thresholds (>= window).
        low_quantile: Quantile below which the LOW regime is entered.
        high_quantile: Quantile above which the HIGH regime is entered.
        periods_per_year: Bars per year for annualisation.

    Returns:
        Integer Series of :class:`VolRegime` codes named ``"vol_regime"``.

    Raises:
        ValueError: If the windows or quantiles are out of range.
    """
    if lookback < window:
        raise ValueError(f"lookback ({lookback}) must be >= window ({window}).")
    if not 0.0 < low_quantile < high_quantile < 1.0:
        raise ValueError(
            f"need 0 < low_quantile < high_quantile < 1, got {low_quantile} and {high_quantile}."
        )

    vol = realized_volatility(returns, window=window, periods_per_year=periods_per_year)
    trailing = vol.rolling(lookback, min_periods=window)
    q_low = trailing.quantile(low_quantile)
    q_mid = trailing.quantile(0.5)
    q_high = trailing.quantile(high_quantile)

    states = _run_state_machine(
        vol.to_numpy(dtype=float),
        q_low.to_numpy(dtype=float),
        q_mid.to_numpy(dtype=float),
        q_high.to_numpy(dtype=float),
    )
    return pd.Series(states, index=returns.index, name="vol_regime")
