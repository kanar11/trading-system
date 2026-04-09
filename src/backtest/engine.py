import pandas as pd


def backtest_strategy(df, transaction_cost=0.001, vol_target=None, vol_window=20):
    df = df.copy()

    if "signal" not in df.columns:
        raise ValueError("DataFrame must contain a 'signal' column.")

    if "close" not in df.columns:
        raise ValueError("DataFrame must contain a 'close' column.")

    # pozycja od następnej sesji
    df["position"] = df["signal"].shift(1).fillna(0)

    # dzienne zwroty rynku
    df["market_returns"] = df["close"].pct_change().fillna(0)

    # volatility targeting
    if vol_target is not None:
        df["realized_vol"] = df["market_returns"].rolling(vol_window).std() * (252 ** 0.5)
        df["vol_scalar"] = vol_target / df["realized_vol"]
        df["vol_scalar"] = df["vol_scalar"].clip(upper=3.0)
        df["vol_scalar"] = df["vol_scalar"].fillna(0)

        df["scaled_position"] = df["position"] * df["vol_scalar"]
    else:
        df["scaled_position"] = df["position"]

    # zwroty strategii przed kosztami
    df["strategy_returns_gross"] = df["scaled_position"] * df["market_returns"]

    # koszt przy zmianie pozycji
    df["trade"] = df["scaled_position"].diff().abs().fillna(0)
    df["transaction_cost"] = df["trade"] * transaction_cost

    # zwroty po kosztach
    df["strategy_returns"] = df["strategy_returns_gross"] - df["transaction_cost"]

    # krzywa kapitału
    df["equity_curve"] = (1 + df["strategy_returns"]).cumprod()

    # prosty trade log oparty o surową pozycję kierunkową
    trade_log = []
    current_position = 0
    entry_date = None
    entry_price = None

    for date, row in df.iterrows():
        new_position = row["position"]
        price = row["close"]

        if current_position == 0 and new_position != 0:
            current_position = new_position
            entry_date = date
            entry_price = price

        elif current_position != 0 and new_position != current_position:
            exit_date = date
            exit_price = price

            if current_position == 1:
                trade_return = (exit_price / entry_price) - 1
            else:
                trade_return = (entry_price / exit_price) - 1

            trade_log.append({
                "entry_date": entry_date,
                "exit_date": exit_date,
                "direction": int(current_position),
                "entry_price": entry_price,
                "exit_price": exit_price,
                "trade_return": trade_return,
                "holding_days": (exit_date - entry_date).days,
            })

            if new_position != 0:
                current_position = new_position
                entry_date = date
                entry_price = price
            else:
                current_position = 0
                entry_date = None
                entry_price = None

    if current_position != 0 and entry_date is not None:
        exit_date = df.index[-1]
        exit_price = df["close"].iloc[-1]

        if current_position == 1:
            trade_return = (exit_price / entry_price) - 1
        else:
            trade_return = (entry_price / exit_price) - 1

        trade_log.append({
            "entry_date": entry_date,
            "exit_date": exit_date,
            "direction": int(current_position),
            "entry_price": entry_price,
            "exit_price": exit_price,
            "trade_return": trade_return,
            "holding_days": (exit_date - entry_date).days,
        })

    trade_log_df = pd.DataFrame(trade_log)

    return df, trade_log_df