import pandas as pd

def build_trade_log(result: pd.DataFrame) -> pd.DataFrame:
    """

    :rtype: object
    """
    df = result.copy()

    pos = df["pos"].astype(int)
    prev_pos = pos.shift(1).fillna(0).astype(int)

    change = pos - prev_pos
    change_dates = df.index[change != 0]

    trades = []

    current_pos = 0
    entry_date = None
    entry_price = None

    for dt in change_dates:
        new_pos = int(pos.loc[dt])
        price = float(df.loc[dt, "close"])

        # close me existing trade !!
        if current_pos != 0 and entry_date is not None:
            exit_date = dt
            exit_price = price

            trade_return = current_pos * (exit_price / entry_price - 1.0)

            trades.append({
                "entry_date": entry_date,
                "exit_date": exit_date,
                "direction": current_pos,
                "entry_price": entry_price,
                "exit_price": exit_price,
                "trade_return": trade_return,
            })

            entry_date = None
            entry_price = None

        # open me new trade
        if new_pos != 0:
            current_pos = new_pos
            entry_date = dt
            entry_price = price
        else:
            current_pos = 0
#close open position at the end (mark-to-market)
    if current_pos != 0 and entry_date is not None:
        exit_date = df.index[-1]
        exit_price = float(df["close"].iloc[-1])
        trade_return = current_pos * (exit_price / entry_price - 1.0)

        trades.append({
            "entry_date": entry_date,
            "exit_date": exit_date,
            "direction": current_pos,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "trade_return": trade_return,

        })
    trade_df = pd.DataFrame(trades)

    if not trade_df.empty:
        trade_df["holding_days"] = (
            trade_df["exit_date"] - trade_df["entry_date"]
        ).dt.days

    return trade_df
