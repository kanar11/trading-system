"""Regression tests: the event engine must enforce time-in-force.

The old implementation never expired anything — DAY, IOC and FOK all
rested forever like GTC. Also covers the STOP_LIMIT gap protection
added alongside (fills at the better of open/limit when the bar opens
through the stop, consistent with plain LIMIT orders).
"""

import pandas as pd

from src.backtest.event_engine import Context, EventEngine
from src.oms import OrderStatus, OrderType, Side, TimeInForce


def _flat_ohlc(closes: list[float]) -> pd.DataFrame:
    """OHLC bars with open=close and a ±1% intrabar range."""
    idx = pd.date_range("2022-01-03", periods=len(closes), freq="B")
    return pd.DataFrame(
        {
            "open": closes,
            "high": [c * 1.01 for c in closes],
            "low": [c * 0.99 for c in closes],
            "close": closes,
        },
        index=idx,
    )


def _ohlc_rows(rows: list[tuple[float, float, float, float]]) -> pd.DataFrame:
    """Explicit (open, high, low, close) bars for gap scenarios."""
    idx = pd.date_range("2022-01-03", periods=len(rows), freq="B")
    return pd.DataFrame(rows, columns=["open", "high", "low", "close"], index=idx)


class SubmitOnce:
    """Submit one preconfigured order on bar 0 and record the status the
    strategy observes on every later bar (mid-bar, during on_bar)."""

    def __init__(
        self,
        side: Side = Side.BUY,
        qty: float = 10,
        order_type: OrderType = OrderType.LIMIT,
        limit_price: float | None = None,
        stop_price: float | None = None,
        tif: TimeInForce = TimeInForce.DAY,
    ):
        self.kwargs = {
            "side": side,
            "quantity": qty,
            "order_type": order_type,
            "limit_price": limit_price,
            "stop_price": stop_price,
            "tif": tif,
        }
        self.order = None
        self.status_seen: dict[int, OrderStatus] = {}
        self._i = -1

    def on_bar(self, ctx: Context) -> None:
        self._i += 1
        if self.order is not None:
            self.status_seen[self._i] = self.order.status
        if self._i == 0:
            self.order = ctx.submit_order(**self.kwargs)


# ---------------------------------------------------------------------------
# DAY expiry
# ---------------------------------------------------------------------------


def test_day_limit_expires_after_its_single_session():
    # buy limit far below the market never fills; its session is bar 1
    strat = SubmitOnce(order_type=OrderType.LIMIT, limit_price=90, tif=TimeInForce.DAY)
    res = EventEngine(symbol="ASSET", initial_cash=10_000).run(_flat_ohlc([100] * 4), strat)

    assert strat.order is not None
    assert strat.order.status is OrderStatus.CANCELLED
    assert res.fills.empty
    # still working while its session (bar 1) is in progress...
    assert strat.status_seen[1] is OrderStatus.WORKING
    # ...cancelled at that bar's close, i.e. before bar 2's on_bar
    assert strat.status_seen[2] is OrderStatus.CANCELLED


def test_gtc_limit_rests_until_touched_bars_later():
    closes = [100, 100, 100, 94, 100]
    day = SubmitOnce(order_type=OrderType.LIMIT, limit_price=95, tif=TimeInForce.DAY)
    gtc = SubmitOnce(order_type=OrderType.LIMIT, limit_price=95, tif=TimeInForce.GTC)

    res_day = EventEngine(symbol="ASSET", initial_cash=10_000).run(_flat_ohlc(closes), day)
    res_gtc = EventEngine(symbol="ASSET", initial_cash=10_000).run(_flat_ohlc(closes), gtc)

    # DAY expired long before the price came down to the limit
    assert res_day.fills.empty
    assert day.order.status is OrderStatus.CANCELLED
    # GTC rested and filled on bar 3 at the open (94, gap-protected)
    assert len(res_gtc.fills) == 1
    assert res_gtc.fills.iloc[0]["price"] == 94.0
    assert gtc.order.status is OrderStatus.FILLED


def test_pre_run_submission_gets_bar0_session_only():
    eng = EventEngine(symbol="ASSET", initial_cash=10_000)
    order = eng.submit_order(
        symbol="ASSET",
        side=Side.BUY,
        quantity=10,
        order_type=OrderType.LIMIT,
        limit_price=90,
        tif=TimeInForce.DAY,
    )

    class Noop:
        def on_bar(self, ctx: Context) -> None:
            pass

    res = eng.run(_flat_ohlc([100, 100, 100]), Noop())
    assert order.status is OrderStatus.CANCELLED
    assert res.fills.empty


# ---------------------------------------------------------------------------
# IOC / FOK
# ---------------------------------------------------------------------------


def test_ioc_unfilled_is_cancelled_immediately_after_matching():
    strat = SubmitOnce(order_type=OrderType.LIMIT, limit_price=90, tif=TimeInForce.IOC)
    res = EventEngine(symbol="ASSET", initial_cash=10_000).run(_flat_ohlc([100] * 3), strat)

    assert res.fills.empty
    assert strat.order.status is OrderStatus.CANCELLED
    # unlike DAY, the cancel happens before bar 1's on_bar even runs
    assert strat.status_seen[1] is OrderStatus.CANCELLED


def test_fok_unfilled_is_cancelled_immediately_after_matching():
    strat = SubmitOnce(order_type=OrderType.LIMIT, limit_price=90, tif=TimeInForce.FOK)
    EventEngine(symbol="ASSET", initial_cash=10_000).run(_flat_ohlc([100] * 3), strat)
    assert strat.order.status is OrderStatus.CANCELLED


def test_ioc_fills_when_marketable_on_its_bar():
    strat = SubmitOnce(order_type=OrderType.LIMIT, limit_price=100, tif=TimeInForce.IOC)
    res = EventEngine(symbol="ASSET", initial_cash=10_000).run(_flat_ohlc([100] * 3), strat)

    assert len(res.fills) == 1
    assert res.fills.iloc[0]["price"] == 100.0
    assert strat.order.status is OrderStatus.FILLED


# ---------------------------------------------------------------------------
# STOP_LIMIT gap protection
# ---------------------------------------------------------------------------


def test_buy_stop_limit_gap_between_stop_and_limit_fills_at_open():
    # bar 1 gaps open at 110, through the 105 stop but below the 115 limit
    df = _ohlc_rows([(100, 101, 99, 100), (110, 111, 109, 110), (110, 111, 109, 110)])
    strat = SubmitOnce(
        order_type=OrderType.STOP_LIMIT, stop_price=105, limit_price=115, tif=TimeInForce.GTC
    )
    res = EventEngine(symbol="ASSET", initial_cash=10_000).run(df, strat)

    assert len(res.fills) == 1
    assert res.fills.iloc[0]["price"] == 110.0


def test_buy_stop_limit_open_above_limit_fills_at_limit_when_reachable():
    # bar 1 opens at 120 (above the 110 limit) but trades down to 108
    df = _ohlc_rows([(100, 101, 99, 100), (120, 121, 108, 112), (112, 113, 111, 112)])
    strat = SubmitOnce(
        order_type=OrderType.STOP_LIMIT, stop_price=105, limit_price=110, tif=TimeInForce.GTC
    )
    res = EventEngine(symbol="ASSET", initial_cash=10_000).run(df, strat)

    assert len(res.fills) == 1
    assert res.fills.iloc[0]["price"] == 110.0


def test_sell_stop_limit_gap_fills_at_the_better_open():
    # bar 1 gaps down through the 95 stop; open 92 is above the 90 limit
    df = _ohlc_rows([(100, 101, 99, 100), (92, 93, 91, 92), (92, 93, 91, 92)])
    strat = SubmitOnce(
        side=Side.SELL,
        order_type=OrderType.STOP_LIMIT,
        stop_price=95,
        limit_price=90,
        tif=TimeInForce.GTC,
    )
    res = EventEngine(symbol="ASSET", initial_cash=10_000).run(df, strat)

    assert len(res.fills) == 1
    assert res.fills.iloc[0]["price"] == 92.0
