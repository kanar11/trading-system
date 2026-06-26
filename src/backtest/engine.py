"""Backtest engine for systematic trading strategies.

Runs a vectorised backtest with transaction costs, optional volatility targeting,
and optional risk management controls. Produces an equity curve and trade log.
"""

import logging
from typing import Any

import numpy as np
import pandas as pd

from src.risk.manager import RiskConfig, apply_risk_controls

logger = logging.getLogger(__name__)


def backtest_strategy(
    df: pd.DataFrame,
    transaction_cost: float = 0.001,
    vol_target: float | None = None,
    vol_window: int = 20,
    risk_config: RiskConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run a full backtest on a DataFrame with trading signals.

    Args:
        df: DataFrame with at least 'close' and 'signal' columns.
        transaction_cost: Round-trip cost as a fraction (e.g. 0.001 = 0.1%).
        vol_target: Annualised volatility target for position scaling.
            Set to None to disable vol targeting.
        vol_window: Lookback window for realised volatility estimate.
        risk_config: Risk management configuration. Pass None to skip
            risk controls entirely, or a RiskConfig instance to enable them.

    Returns:
        Tuple of (backtest_df, trade_log_df).

    Raises:
        ValueError: If required columns are missing.
    """
    if "signal" not in df.columns:
        raise ValueError("DataFrame must contain a 'signal' column.")
    if "close" not in df.columns:
        raise ValueError("DataFrame must contain a 'close' column.")

    df = df.copy()

    # position enters on the next bar
    df["position"] = df["signal"].shift(1).fillna(0)

    # daily market returns
    df["market_returns"] = df["close"].pct_change().fillna(0)

    # volatility targeting
    if vol_target is not None:
        df["realized_vol"] = df["market_returns"].rolling(vol_window).std() * np.sqrt(252)
        df["vol_scalar"] = (vol_target / df["realized_vol"]).clip(upper=3.0).fillna(0)
        df["scaled_position"] = df["position"] * df["vol_scalar"]
    else:
        df["scaled_position"] = df["position"]

    # --- risk management layer ---
    if risk_config is not None:
        df = apply_risk_controls(df, config=risk_config)

    # gross strategy returns
    df["strategy_returns_gross"] = df["scaled_position"] * df["market_returns"]

    # transaction costs on position changes
    df["trade"] = df["scaled_position"].diff().abs().fillna(0)
    df["transaction_cost"] = df["trade"] * transaction_cost

    # net strategy returns
    df["strategy_returns"] = df["strategy_returns_gross"] - df["transaction_cost"]

    # equity curve
    df["equity_curve"] = (1 + df["strategy_returns"]).cumprod()

    # build trade log
    trade_log_df = _build_trade_log(df)

    return df, trade_log_df


def _build_trade_log(df: pd.DataFrame) -> pd.DataFrame:
    """Extract a trade-level log from the backtest DataFrame.

    Each row represents a single round-trip trade with entry/exit dates,
    prices, direction, return, and holding period. An open position at the
    end of the series is marked to market on the final bar.
    """
    positions = df["position"].to_numpy()
    prices = df["close"].to_numpy()
    index = pd.DatetimeIndex(df.index)
    n = len(df)

    trades: list[dict[str, Any]] = []
    current_position = 0.0
    entry_idx: int | None = None

    for i in range(n):
        new_position = float(positions[i])

        # new entry from flat
        if current_position == 0 and new_position != 0:
            current_position = new_position
            entry_idx = i

        # position change or exit (including a direct flip)
        elif current_position != 0 and new_position != current_position:
            if entry_idx is not None:
                trades.append(_make_trade(index, prices, entry_idx, i, current_position))
            if new_position != 0:
                current_position = new_position
                entry_idx = i
            else:
                current_position = 0.0
                entry_idx = None

    # mark-to-market open position at end
    if current_position != 0 and entry_idx is not None:
        trades.append(_make_trade(index, prices, entry_idx, n - 1, current_position))

    return pd.DataFrame(trades)


def _make_trade(
    index: pd.DatetimeIndex,
    prices: np.ndarray,
    entry_idx: int,
    exit_idx: int,
    direction: float,
) -> dict[str, Any]:
    """Assemble a single round-trip trade record from positional indices."""
    entry_price = float(prices[entry_idx])
    exit_price = float(prices[exit_idx])
    if direction > 0:
        trade_return = exit_price / entry_price - 1.0
    else:
        trade_return = entry_price / exit_price - 1.0

    return {
        "entry_date": index[entry_idx],
        "exit_date": index[exit_idx],
        "direction": int(direction),
        "entry_price": entry_price,
        "exit_price": exit_price,
        "trade_return": trade_return,
        "holding_days": (index[exit_idx] - index[entry_idx]).days,
    }
