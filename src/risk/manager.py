"""Risk management module for systematic trading strategies.

Provides position-level and portfolio-level risk controls:
- stop-loss and take-profit thresholds
- trailing stop based on peak equity since entry
- maximum position size cap
- daily loss limit (circuit breaker)
"""

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class RiskConfig:
    """Configuration for all risk management rules.

    Attributes:
        stop_loss: Maximum loss per trade before forced exit (e.g. 0.05 = 5%).
            Set to None to disable.
        take_profit: Target profit per trade for automatic exit (e.g. 0.10 = 10%).
            Set to None to disable.
        trailing_stop: Drawdown from peak equity since entry that triggers exit
            (e.g. 0.03 = 3%). Set to None to disable.
        max_position: Maximum absolute position size allowed (e.g. 1.0 = 100%).
            Positions exceeding this are clipped.
        daily_loss_limit: Maximum cumulative loss in a single day before
            all positions are flattened (e.g. 0.02 = 2%). Set to None to disable.
    """

    stop_loss: float | None = 0.05
    take_profit: float | None = 0.10
    trailing_stop: float | None = 0.03
    max_position: float = 1.0
    daily_loss_limit: float | None = 0.02


def apply_risk_controls(
    df: pd.DataFrame,
    config: RiskConfig | None = None,
) -> pd.DataFrame:
    """Apply risk management rules to a DataFrame that already has positions.

    Expects columns: 'close', 'position' (raw directional), 'scaled_position'.
    Modifies 'scaled_position' in place and adds diagnostic columns.

    Execution convention (look-ahead-free): every rule is *decided* on a
    bar's close and *executed* on the next bar, mirroring the engine's
    ``position = signal.shift(1)`` convention. The bar whose close breaches
    a stop therefore keeps its position (and suffers that bar's loss);
    the position is flat from the following bar. ``risk_event`` is flagged
    on the decision bar. A rule that fires on the very last bar has no
    execution bar left, so only the flag is recorded.

    Entry prices are taken at the previous bar's close (the fill price
    implied by the shift-by-one convention). The daily-loss circuit
    breaker accrues each bar's P&L with the position actually held over
    that bar and, once breached, keeps the book flat for the remainder
    of that calendar day.

    Args:
        df: Backtest DataFrame with price data and position columns.
        config: Risk configuration. Uses defaults if None.

    Returns:
        DataFrame with adjusted positions and risk event flags.
    """
    if config is None:
        config = RiskConfig()

    df = df.copy()
    prices = df["close"].to_numpy(dtype=float)
    raw_position = df["scaled_position"].to_numpy(dtype=float).copy()
    n = len(df)

    # output arrays
    adjusted_position = raw_position.copy()
    risk_event = np.full(n, "", dtype=object)

    # tracking state
    entry_price: float | None = None
    peak_price: float | None = None
    prev_pos_dir: float = 0.0
    daily_pnl: float = 0.0
    prev_date = None
    circuit_breaker_active = False
    force_flat = False  # exit decided on the previous close, executes on this bar

    for i in range(n):
        price = prices[i]
        current_date = df.index[i].date() if hasattr(df.index[i], "date") else None

        # --- reset daily PnL / release circuit breaker on a new day ---
        if current_date is not None and current_date != prev_date:
            daily_pnl = 0.0
            circuit_breaker_active = False
            prev_date = current_date

        # --- execute an exit decided at the previous bar's close ---
        if force_flat:
            adjusted_position[i] = 0.0
            force_flat = False
            prev_pos_dir = 0.0
            continue

        # --- circuit breaker: flat for the rest of the day ---
        if circuit_breaker_active:
            adjusted_position[i] = 0.0
            risk_event[i] = "daily_limit"
            prev_pos_dir = 0.0
            continue

        pos = raw_position[i]
        pos_dir = float(np.sign(pos))

        # --- max position size cap (applies every bar) ---
        if abs(pos) > config.max_position:
            pos = config.max_position * pos_dir
            adjusted_position[i] = pos
            risk_event[i] = "pos_capped"

        # --- detect new trade entry (fill at the previous bar's close) ---
        if pos_dir != 0 and pos_dir != prev_pos_dir:
            entry_price = prices[i - 1] if i > 0 else price
            peak_price = entry_price
        elif pos_dir == 0:
            entry_price = None
            peak_price = None

        # --- update the favourable extreme for the trailing stop ---
        if peak_price is not None and pos_dir != 0:
            peak_price = max(peak_price, price) if pos_dir > 0 else min(peak_price, price)

        # --- accrue today's P&L with the position held over this bar ---
        if i > 0:
            bar_ret = (prices[i] - prices[i - 1]) / prices[i - 1]
            daily_pnl += adjusted_position[i] * bar_ret

        # --- decisions at this bar's close (executed next bar) ---
        triggered = ""
        if pos_dir != 0 and entry_price is not None:
            if pos_dir > 0:
                unrealised = (price - entry_price) / entry_price
            else:
                unrealised = (entry_price - price) / entry_price

            if config.stop_loss is not None and unrealised <= -config.stop_loss:
                triggered = "stop_loss"
            elif config.take_profit is not None and unrealised >= config.take_profit:
                triggered = "take_profit"
            elif config.trailing_stop is not None and peak_price is not None:
                if pos_dir > 0:
                    drawdown = (peak_price - price) / peak_price
                else:
                    drawdown = (price - peak_price) / peak_price
                if drawdown >= config.trailing_stop:
                    triggered = "trailing_stop"

        if config.daily_loss_limit is not None and daily_pnl <= -config.daily_loss_limit:
            triggered = "daily_limit"
            circuit_breaker_active = True

        if triggered:
            # exit events take precedence over a same-bar "pos_capped" flag
            risk_event[i] = triggered
            force_flat = True
            entry_price = None
            peak_price = None

        prev_pos_dir = pos_dir

    df["scaled_position"] = adjusted_position
    df["risk_event"] = risk_event

    return df


def summarise_risk_events(df: pd.DataFrame) -> dict[str, int]:
    """Count how many times each risk event type was triggered.

    Args:
        df: DataFrame returned by apply_risk_controls.

    Returns:
        Dictionary mapping event type to count.
    """
    if "risk_event" not in df.columns:
        return {}

    events = df["risk_event"]
    events = events[events != ""]
    return dict(events.value_counts())
