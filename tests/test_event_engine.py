"""Tests for the event-driven backtest engine."""

import numpy as np
import pandas as pd
import pytest

from src.backtest.event_engine import Context, EventEngine, EventEngineResult
from src.oms import OrderStatus, OrderType, Side


def _ohlc(prices: list[float]) -> pd.DataFrame:
    """Build a minimal OHLC frame from a list of closes."""
    closes = np.array(prices, dtype=float)
    dates = pd.date_range("2020-01-01", periods=len(closes), freq="B")
    return pd.DataFrame(
        {
            "open": closes,
            "high": closes * 1.01,
            "low": closes * 0.99,
            "close": closes,
        },
        index=dates,
    )


# ---------------------------------------------------------------------------
# strategies used as fixtures
# ---------------------------------------------------------------------------


class BuyOnceStrategy:
    """Submit a single market BUY on the first bar; sit tight after."""

    def __init__(self, qty: float = 10):
        self.qty = qty
        self._submitted = False

    def on_bar(self, ctx: Context) -> None:
        if not self._submitted:
            ctx.submit_order(Side.BUY, self.qty)
            self._submitted = True


class BuyAndExitStrategy:
    """Buy on bar 0, sell at limit_price on every subsequent bar."""

    def __init__(self, qty: float = 10, exit_at: float = 110):
        self.qty = qty
        self.exit_at = exit_at
        self._bought = False
        self._exit_submitted = False

    def on_bar(self, ctx: Context) -> None:
        if not self._bought:
            ctx.submit_order(Side.BUY, self.qty)
            self._bought = True
        elif not self._exit_submitted:
            ctx.submit_order(
                Side.SELL,
                self.qty,
                order_type=OrderType.LIMIT,
                limit_price=self.exit_at,
            )
            self._exit_submitted = True


class StopLossStrategy:
    """Buy on bar 0, set a STOP sell below the entry."""

    def __init__(self, qty: float = 10, stop: float = 95):
        self.qty = qty
        self.stop = stop
        self._bought = False
        self._stop_submitted = False

    def on_bar(self, ctx: Context) -> None:
        if not self._bought:
            ctx.submit_order(Side.BUY, self.qty)
            self._bought = True
        elif not self._stop_submitted:
            ctx.submit_order(
                Side.SELL,
                self.qty,
                order_type=OrderType.STOP,
                stop_price=self.stop,
            )
            self._stop_submitted = True


class DoNothingStrategy:
    def on_bar(self, ctx: Context) -> None:
        pass


# ---------------------------------------------------------------------------
# engine sanity
# ---------------------------------------------------------------------------


def test_engine_runs_with_no_trades():
    df = _ohlc([100, 101, 102, 103])
    eng = EventEngine(symbol="ASSET", initial_cash=10_000)
    res = eng.run(df, DoNothingStrategy())
    assert isinstance(res, EventEngineResult)
    assert len(res.equity_curve) == 4
    assert res.equity_curve.iloc[-1] == 10_000
    assert res.fills.empty


def test_engine_validates_columns():
    df = pd.DataFrame({"close": [100, 101]})
    with pytest.raises(ValueError, match="missing"):
        EventEngine().run(df, DoNothingStrategy())


# ---------------------------------------------------------------------------
# market order behaviour
# ---------------------------------------------------------------------------


def test_market_buy_fills_on_next_bar_open():
    df = _ohlc([100, 105, 110])
    eng = EventEngine(symbol="ASSET", initial_cash=10_000)
    res = eng.run(df, BuyOnceStrategy(qty=10))
    # order is submitted on bar 0; matched on bar 1's open (105)
    fills = res.fills
    assert len(fills) == 1
    assert fills.iloc[0]["price"] == pytest.approx(105.0)
    # quantity is full
    assert res.portfolio.positions["ASSET"].quantity == 10


def test_market_buy_equity_accounting():
    df = _ohlc([100, 100, 100])
    eng = EventEngine(symbol="ASSET", initial_cash=10_000)
    res = eng.run(df, BuyOnceStrategy(qty=10))
    # cash 10_000 - 10*100 = 9000; equity unchanged at 10_000
    assert res.portfolio.cash == pytest.approx(9000)
    assert res.equity_curve.iloc[-1] == pytest.approx(10_000)


def test_commission_charged():
    df = _ohlc([100, 100, 100])
    eng = EventEngine(
        symbol="ASSET",
        initial_cash=10_000,
        commission_per_share=0.01,
        commission_min=1.0,
    )
    res = eng.run(df, BuyOnceStrategy(qty=10))
    # commission = max(10 * 0.01, 1.0) = 1.0
    assert res.portfolio.fees_paid == pytest.approx(1.0)
    assert res.portfolio.cash == pytest.approx(10_000 - 10 * 100 - 1.0)


def test_slippage_increases_buy_price():
    df = _ohlc([100, 100, 100])
    eng = EventEngine(symbol="ASSET", initial_cash=10_000, slippage_bps=50)
    res = eng.run(df, BuyOnceStrategy(qty=10))
    # 50 bps = 0.5% adverse on buy
    assert res.fills.iloc[0]["price"] == pytest.approx(100 * 1.005)


# ---------------------------------------------------------------------------
# limit order behaviour
# ---------------------------------------------------------------------------


def test_limit_sell_fills_when_high_touches_limit():
    # bar prices: 100 → 105 (high 106.05) → 115 (high 116.15)
    df = _ohlc([100, 105, 115])
    eng = EventEngine(symbol="ASSET", initial_cash=10_000)
    res = eng.run(df, BuyAndExitStrategy(qty=10, exit_at=110))
    # Bar 0: BUY market submitted
    # Bar 1: BUY fills at open=105. Sell LIMIT@110 submitted.
    # Bar 2: high = 115*1.01 = 116.15 ≥ 110 → fills at 110 (gap-protected by open=115)
    fills = res.fills
    assert len(fills) == 2
    sell = fills.iloc[1]
    assert sell["side"] == "SELL"
    # When the bar opens already above the limit (115 > 110), the engine fills
    # at the open price for sells (favorable to seller)
    assert sell["price"] >= 110


def test_limit_does_not_fill_if_price_out_of_reach():
    df = _ohlc([100, 100, 100, 100])
    eng = EventEngine(symbol="ASSET", initial_cash=10_000)
    res = eng.run(df, BuyAndExitStrategy(qty=10, exit_at=200))
    # sell limit at 200 never reached
    sells = res.fills[res.fills["side"] == "SELL"]
    assert sells.empty


# ---------------------------------------------------------------------------
# stop order behaviour
# ---------------------------------------------------------------------------


def test_stop_loss_triggers_on_breach():
    # closes: 100, 100, 90, 85 — bar 2 has low=90*0.99=89.1, triggers stop@95
    df = _ohlc([100, 100, 90, 85])
    eng = EventEngine(symbol="ASSET", initial_cash=10_000)
    res = eng.run(df, StopLossStrategy(qty=10, stop=95))
    sells = res.fills[res.fills["side"] == "SELL"]
    assert not sells.empty
    # filled at min(open=90, stop=95) = 90 for sell stop (gap down)
    assert sells.iloc[0]["price"] <= 95


def test_stop_does_not_trigger_above_stop():
    df = _ohlc([100, 100, 100, 100])
    eng = EventEngine(symbol="ASSET", initial_cash=10_000)
    res = eng.run(df, StopLossStrategy(qty=10, stop=95))
    sells = res.fills[res.fills["side"] == "SELL"]
    assert sells.empty


# ---------------------------------------------------------------------------
# state tracking
# ---------------------------------------------------------------------------


def test_equity_curve_reflects_unrealised_pnl():
    df = _ohlc([100, 100, 120, 120])
    eng = EventEngine(symbol="ASSET", initial_cash=10_000)
    res = eng.run(df, BuyOnceStrategy(qty=10))
    # after fill at open of bar 1, we own 10 shares cost 100
    # close of bar 2 = 120 → unrealised = 200 → equity = 10_200
    assert res.equity_curve.iloc[-1] == pytest.approx(10_200)


def test_orders_are_terminal_after_full_fill():
    df = _ohlc([100, 105])
    eng = EventEngine(symbol="ASSET", initial_cash=10_000)
    res = eng.run(df, BuyOnceStrategy(qty=10))
    o = res.orders[0]
    assert o.status is OrderStatus.FILLED
    assert o.filled_quantity == 10


def test_cancel_all_returns_count():
    df = _ohlc([100, 100, 100])
    eng = EventEngine(symbol="ASSET", initial_cash=10_000)

    class CancellerStrategy:
        def __init__(self):
            self._step = 0

        def on_bar(self, ctx):
            if self._step == 0:
                ctx.submit_order(
                    Side.BUY,
                    10,
                    order_type=OrderType.LIMIT,
                    limit_price=50,
                )  # never fills
            if self._step == 1:
                ctx.cancel_all()
            self._step += 1

    res = eng.run(df, CancellerStrategy())
    o = res.orders[0]
    assert o.status is OrderStatus.CANCELLED
