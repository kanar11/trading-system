"""Risk management module for systematic trading strategies.

Provides position-level and portfolio-level risk controls:
- stop-loss and take-profit thresholds
- trailing stop based on peak equity since entry
- maximum position size cap
- daily loss limit (circuit breaker)
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd


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

    Args:
        df: Backtest DataFrame with price data and position columns.
        config: Risk configuration. Uses defaults if None.

    Returns:
        DataFrame with adjusted positions and risk event flags.
    """
    if config is None:
        config = RiskConfig()

    df = df.copy()
    prices = df["close"].values
    raw_position = df["scaled_position"].values.copy()
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

    for i in range(n):
        price = prices[i]
        pos = raw_position[i]
        pos_dir = np.sign(pos)
        current_date = df.index[i].date() if hasattr(df.index[i], "date") else None

        # --- reset daily PnL on new day ---
        if current_date is not None and current_date != prev_date:
            daily_pnl = 0.0
            circuit_breaker_active = False
            prev_date = current_date

        # --- circuit breaker: daily loss limit ---
        if circuit_breaker_active:
            adjusted_position[i] = 0.0
            risk_event[i] = "daily_limit"
            continue

        # --- detect new trade entry ---
        if pos_dir != 0 and pos_dir != prev_pos_dir:
            entry_price = price
            peak_price = price

        # --- update peak price for trailing stop ---
        if entry_price is not None and pos_dir != 0:
            if pos_dir > 0:
                peak_price = max(peak_price, price)
            else:
                peak_price = min(peak_price, price)

        # --- check stop-loss ---
        if (
            config.stop_loss is not None
            and entry_price is not None
            and pos_dir != 0
        ):
            if pos_dir > 0:
                unrealised = (price - entry_price) / entry_price
            else:
                unrealised = (entry_price - price) / entry_price

            if unrealised <= -config.stop_loss:
                adjusted_position[i] = 0.0
                risk_event[i] = "stop_loss"
                entry_price = None
                peak_price = None
                prev_pos_dir = 0.0
                continue

        # --- check take-profit ---
        if (
            config.take_profit is not None
            and entry_price is not None
            and pos_dir != 0
        ):
            if pos_dir > 0:
                unrealised = (price - entry_price) / entry_price
            else:
                unrealised = (entry_price - price) / entry_price

            if unrealised >= config.take_profit:
                adjusted_position[i] = 0.0
                risk_event[i] = "take_profit"
                entry_price = None
                peak_price = None
                prev_pos_dir = 0.0
                continue

        # --- check trailing stop ---
        if (
            config.trailing_stop is not None
            and peak_price is not None
            and pos_dir != 0
        ):
            if pos_dir > 0:
                drawdown = (peak_price - price) / peak_price
            else:
                drawdown = (price - peak_price) / peak_price

            if drawdown >= config.trailing_stop:
                adjusted_position[i] = 0.0
                risk_event[i] = "trailing_stop"
                entry_price = None
                peak_price = None
                prev_pos_dir = 0.0
                continue

        # --- max position size cap ---
        if abs(pos) > config.max_position:
            adjusted_position[i] = config.max_position * pos_dir
            risk_event[i] = "pos_capped"

        # --- track daily PnL for circuit breaker ---
        if i > 0 and adjusted_position[i] != 0:
            ret = (prices[i] - prices[i - 1]) / prices[i - 1]
            daily_pnl += adjusted_position[i - 1] * ret

            if config.daily_loss_limit is not None and daily_pnl <= -config.daily_loss_limit:
                adjusted_position[i] = 0.0
                risk_event[i] = "daily_limit"
                circuit_breaker_active = True
                entry_price = None
                peak_price = None
                prev_pos_dir = 0.0
                continue

        prev_pos_dir = np.sign(adjusted_position[i])

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
