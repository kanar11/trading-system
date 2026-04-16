"""Backtest engine for systematic trading strategies.

Runs a vectorised backtest with transaction costs, optional volatility targeting,
and optional risk management controls. Produces an equity curve and trade log.
"""

import pandas as pd
import numpy as np

from src.risk.manager import RiskConfig, apply_risk_controls


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
        df["realized_vol"] = (
            df["market_returns"].rolling(vol_window).std() * np.sqrt(252)
        )
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
    prices, direction, return, and holding period.
    """
    trades: list[dict] = []
    current_position: float = 0.0
    entry_date = None
    entry_price: float | None = None

    for date, row in df.iterrows():
        new_position = row["position"]
        price = row["close"]

        # new entry from flat
        if current_position == 0 and new_position != 0:
            current_position = new_position
            entry_date = date
            entry_price = price

        # position change or exit
        elif current_position != 0 and new_position != current_position:
            if entry_price is not None:
                trade_return = (
                    (price / entry_price) - 1
                    if current_position == 1
                    else (entry_price / price) - 1
                )
                trades.append(
                    {
                        "entry_date": entry_date,
                        "exit_date": date,
                        "direction": int(current_position),
                        "entry_price": entry_price,
                        "exit_price": price,
                        "trade_return": trade_return,
                        "holding_days": (date - entry_date).days,
                    }
                )

            # flip to new position
            if new_position != 0:
                current_position = new_position
                entry_date = date
                entry_price = price
            else:
                current_position = 0
                entry_date = None
                entry_price = None

    # mark-to-market open position at end
    if current_position != 0 and entry_price is not None:
        last_date = df.index[-1]
        last_price = df["close"].iloc[-1]
        trade_return = (
            (last_price / entry_price) - 1
            if current_position == 1
            else (entry_price / last_price) - 1
        )
        trades.append(
            {
                "entry_date": entry_date,
                "exit_date": last_date,
                "direction": int(current_position),
                "entry_price": entry_price,
                "exit_price": last_price,
                "trade_return": trade_return,
                "holding_days": (last_date - entry_date).days,
            }
        )

    return pd.DataFrame(trades)
