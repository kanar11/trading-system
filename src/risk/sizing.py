"""Position-sizing helpers.

Self-contained sizing routines that take a return / trade history and
produce a position-size multiplier in the range [0, max_size]. They are
independent of the backtest engine so they can be used at calibration
time (e.g. fit a Kelly fraction on in-sample data, apply it on OOS) or
as a post-hoc study against an existing trade log.

Sizing styles provided:
    - kelly_fraction:         Edge / variance based fractional Kelly.
    - atr_position_size:      Cap notional so loss-per-stop = risk budget.
    - fixed_fractional:       Risk a fixed fraction of equity per trade.
    - volatility_target_size: Scale exposure toward a target volatility.
    - cppi_fraction:          CPPI risky exposure above a protective floor.
    - drawdown_throttle:      De-risk as drawdown approaches a tolerated cap.
    - optimal_f:              Ralph Vince f maximising geometric growth.
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


def volatility_target_size(
    realized_vol: float,
    target_vol: float = 0.15,
    max_size: float = 1.0,
) -> float:
    """Scale exposure so realised volatility matches a target.

    fraction = target_vol / realized_vol, clamped to [0, max_size]. A more
    volatile asset gets a smaller position. Returns 0.0 when either volatility
    is non-positive (no usable risk estimate).

    Args:
        realized_vol: Estimated (annualised) volatility of the asset/strategy.
        target_vol: Desired portfolio volatility (same units as realized_vol).
        max_size: Hard cap on the returned size (limits leverage).

    Returns:
        Position size in [0, max_size].
    """
    if realized_vol <= 0 or target_vol <= 0:
        return 0.0
    return float(min(target_vol / realized_vol, max_size))


def cppi_fraction(
    equity: float,
    floor: float,
    multiplier: float = 3.0,
    max_size: float = 1.0,
) -> float:
    """Constant Proportion Portfolio Insurance risky-asset exposure.

    exposure = multiplier * (equity - floor) / equity, clamped to
    [0, max_size]. As equity falls toward the protective ``floor`` the cushion
    shrinks and exposure de-risks automatically; at or below the floor it is 0.

    Args:
        equity: Current account equity.
        floor: Protective floor below which no risk is taken.
        multiplier: CPPI multiplier on the cushion (equity - floor).
        max_size: Hard cap on the returned size.

    Returns:
        Risky-asset fraction in [0, max_size].
    """
    if equity <= 0 or multiplier <= 0:
        return 0.0
    cushion = equity - floor
    if cushion <= 0:
        return 0.0
    return float(min(multiplier * cushion / equity, max_size))


def drawdown_throttle(
    current_drawdown: float,
    max_drawdown: float,
    max_size: float = 1.0,
) -> float:
    """Linearly de-risk as the current drawdown approaches a tolerated max.

    size = max_size * (1 - |current_drawdown| / max_drawdown), clamped to
    [0, max_size]. At zero drawdown returns ``max_size``; at or beyond
    ``max_drawdown`` returns 0.

    Args:
        current_drawdown: Current drawdown (sign-insensitive; e.g. -0.08 or 0.08).
        max_drawdown: Maximum tolerated drawdown magnitude (> 0).
        max_size: Exposure at zero drawdown.

    Returns:
        Throttled position size in [0, max_size].
    """
    if max_drawdown <= 0:
        return 0.0
    scale = 1.0 - abs(current_drawdown) / max_drawdown
    return float(np.clip(scale, 0.0, 1.0) * max_size)


def optimal_f(
    trades: pd.Series | np.ndarray | list[float],
    cap: float = 1.0,
    resolution: int = 1000,
) -> float:
    """Ralph Vince optimal ``f`` from a set of trade outcomes.

    Finds the fraction ``f`` in (0, 1) that maximises the Terminal Wealth
    Relative ``prod(1 + f * trade_i / |worst_loss|)`` — i.e. the geometric
    growth rate of capital risked per trade.

    Args:
        trades: Per-trade P&L or returns (sign matters; losses negative).
        cap: Hard cap on the returned fraction.
        resolution: Number of grid points searched over (0, 1).

    Returns:
        Optimal ``f`` in [0, cap]. Returns 0.0 with no positive expectancy or
        no data, and ``cap`` when there are no losing trades (growth unbounded).
    """
    t = np.asarray(pd.Series(trades).dropna(), dtype=float)
    if t.size == 0:
        return 0.0

    worst = -float(t.min())
    if worst <= 0:  # no losing trade -> TWR grows without bound in f
        return float(cap)
    if float(t.mean()) <= 0:  # no edge -> risk nothing
        return 0.0

    fs = np.linspace(1e-4, 1.0 - 1e-4, resolution)
    ratios = t / worst
    log_twr = np.log1p(np.outer(fs, ratios)).sum(axis=1)
    best_f = float(fs[int(np.argmax(log_twr))])
    return float(min(best_f, cap))
