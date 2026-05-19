"""Position-sizing helpers.

Self-contained sizing routines that take a return / trade history and
produce a position-size multiplier in the range [0, max_size]. They are
independent of the backtest engine so they can be used at calibration
time (e.g. fit a Kelly fraction on in-sample data, apply it on OOS) or
as a post-hoc study against an existing trade log.

Three sizing styles are provided:
    - kelly_fraction:        Edge / variance based fractional Kelly.
    - atr_position_size:     Cap notional so loss-per-stop = risk budget.
    - fixed_fractional:      Risk a fixed fraction of equity per trade.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def kelly_fraction(
    returns: pd.Series,
    cap: float = 1.0,
    kelly_fraction_of_full: float = 0.5,
) -> float:
    """Estimate a (fractional) Kelly position size from a return series.

    Uses the continuous Kelly approximation f* = mean / variance, which
    is appropriate for log-returns. Full Kelly is famously aggressive,
    so this function applies a multiplier (default half-Kelly) and a
    hard cap.

    Args:
        returns: Per-trade or per-bar return series (e.g. trade_log['trade_return']).
        cap: Maximum position size allowed (e.g. 1.0 = 100% notional).
        kelly_fraction_of_full: Fraction of full Kelly to apply
            (0.5 = half-Kelly, a standard conservative choice).

    Returns:
        Position size in [0, cap]. Returns 0.0 when the edge is
        non-positive or the input is empty.
    """
    r = pd.Series(returns).dropna()
    if len(r) < 2:
        return 0.0

    mean = float(r.mean())
    var = float(r.var(ddof=1))
    if var <= 0 or mean <= 0:
        return 0.0

    full_kelly = mean / var
    sized = kelly_fraction_of_full * full_kelly
    return float(np.clip(sized, 0.0, cap))


def atr_position_size(
    price: float,
    atr: float,
    equity: float,
    risk_per_trade: float = 0.01,
    atr_multiple: float = 2.0,
    max_size: float = 1.0,
) -> float:
    """Volatility-based sizing: notional such that a stop = ``risk_per_trade`` of equity.

    Stop distance = ``atr_multiple`` * ATR. The position size (fraction
    of equity) is chosen so that hitting the stop costs exactly
    ``risk_per_trade`` of equity.

    Args:
        price: Current price.
        atr: Current ATR value (same units as price).
        equity: Current account equity (only its scale matters — the
            result is a fraction).
        risk_per_trade: Fraction of equity to risk on the stop
            (e.g. 0.01 = 1%).
        atr_multiple: Multiples of ATR for the stop distance.
        max_size: Hard cap on the returned size.

    Returns:
        Position size as a fraction of equity, in [0, max_size].
    """
    if price <= 0 or atr <= 0 or equity <= 0:
        return 0.0

    stop_distance = atr_multiple * atr
    dollar_risk = risk_per_trade * equity
    # units = dollar_risk / stop_distance; notional = units * price
    notional = (dollar_risk / stop_distance) * price
    fraction = notional / equity
    return float(min(fraction, max_size))


def fixed_fractional(
    win_rate: float,
    payoff_ratio: float,
    cap: float = 1.0,
) -> float:
    """Discrete-outcome Kelly fraction from win-rate and payoff.

    Useful when sizing is calibrated from trade-log statistics rather
    than from raw returns. f* = W - (1 - W) / R where W = win-rate
    and R = avg-win / |avg-loss|.

    Args:
        win_rate: Probability of a winning trade (0..1).
        payoff_ratio: Average win / |average loss|.
        cap: Hard cap on the returned size.

    Returns:
        Position size in [0, cap]. Returns 0.0 when the edge is
        non-positive.
    """
    if not 0 < win_rate < 1 or payoff_ratio <= 0:
        return 0.0

    f = win_rate - (1 - win_rate) / payoff_ratio
    return float(np.clip(f, 0.0, cap))
