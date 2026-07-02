"""Event-driven backtest engine.

A bar-by-bar simulator that walks a strategy through OHLCV data,
matches pending orders against intrabar prices, and updates the OMS
:class:`Portfolio`. Designed to complement (not replace) the existing
vectorised engine in ``src.backtest.engine``:

    * **Vectorised engine** — fast, signal-in / equity-out. Great for
      research, parameter sweeps, and walk-forward. Doesn't model
      order types, fill ordering, or partial fills.

    * **Event engine** — slower but realistic. Supports MARKET,
      LIMIT, STOP and STOP_LIMIT orders, charges commission +
      slippage per fill, tracks per-symbol positions with cost
      basis. Same `Portfolio` API as live trading.

Fill model:
    * MARKET orders fill at the *next* bar's open (look-ahead-free).
    * LIMIT orders fill at the limit price if the bar's price range
      ([low, high]) touches it.
    * STOP orders convert to MARKET when triggered, filling at
      next-bar open (gap-safe).
    * STOP_LIMIT triggers at stop_price and then behaves like a LIMIT
      (with the same gap protection: a bar that opens through the stop
      fills at the better of open/limit).

Time-in-force:
    * GTC orders rest until filled or cancelled.
    * DAY orders live for exactly one matching session (the bar after
      submission — this engine treats one bar as one day) and are
      cancelled at that bar's close if still unfilled.
    * IOC / FOK orders get a single matching attempt and are cancelled
      immediately after it. Fills here are always all-or-nothing for
      the remaining quantity, so IOC and FOK coincide.

Slippage is applied as a fraction of the fill price in the
unfavourable direction (buys pay more, sells receive less).
"""

from __future__ import annotations

import itertools
import logging
from dataclasses import dataclass, field
from typing import Protocol

import pandas as pd

from src.oms import Liquidity, Order, OrderStatus, OrderType, Portfolio, Side, TimeInForce

logger = logging.getLogger(__name__)


class StrategyProtocol(Protocol):
    """Minimal protocol the event engine requires from a strategy.

    The engine calls ``on_bar`` once per bar with a ``Context`` that
    exposes the live :class:`Portfolio`, the most recent OHLCV row,
    and helpers to submit / cancel orders. Strategies that don't need
    the full protocol can just implement ``on_bar``.
    """

    def on_bar(self, ctx: Context) -> None: ...


@dataclass
class Context:
    """Per-bar context passed to ``strategy.on_bar``.

    Attributes:
        ts: Timestamp of the current bar.
        bar: OHLCV row for the current bar (as a Series).
        history: All bars seen *up to and including* the current bar.
        portfolio: Live OMS Portfolio.
        engine: Reference back to the engine for submitting orders.
        symbol: The traded symbol (single-asset engine convention).
    """

    ts: pd.Timestamp
    bar: pd.Series
    history: pd.DataFrame
    portfolio: Portfolio
    engine: EventEngine
    symbol: str

    def submit_order(
        self,
        side: Side,
        quantity: float,
        order_type: OrderType = OrderType.MARKET,
        limit_price: float | None = None,
        stop_price: float | None = None,
        tif: TimeInForce = TimeInForce.DAY,
        client_tag: str | None = None,
    ) -> Order:
        return self.engine.submit_order(
            symbol=self.symbol,
            side=side,
            quantity=quantity,
            order_type=order_type,
            limit_price=limit_price,
            stop_price=stop_price,
            tif=tif,
            client_tag=client_tag,
        )

    def cancel_all(self) -> int:
        return self.engine.cancel_all(self.symbol)


@dataclass
class EventEngineResult:
    """Output of :meth:`EventEngine.run`.

    Attributes:
        equity_curve: Series of mark-to-market equity, one point per bar.
        returns: Daily returns derived from ``equity_curve``.
        portfolio: Final Portfolio state (positions + cash + fees).
        orders: All orders that ever existed (terminal and active).
        fills: Wide-format fill log: ts, symbol, side, qty, price, commission.
    """

    equity_curve: pd.Series
    returns: pd.Series
    portfolio: Portfolio
    orders: list[Order]
    fills: pd.DataFrame


@dataclass
class EventEngine:
    """Single-asset event-driven backtest engine.

    Attributes:
        symbol: The instrument being traded.
        initial_cash: Starting cash for the portfolio.
        commission_per_share: Flat per-share commission, in account currency.
        commission_min: Minimum commission per order (when computed > 0).
        slippage_bps: Adverse slippage applied to every fill (in basis points).
    """

    symbol: str = "ASSET"
    initial_cash: float = 100_000.0
    commission_per_share: float = 0.0
    commission_min: float = 0.0
    slippage_bps: float = 0.0
    _next_order_id: itertools.count = field(default_factory=lambda: itertools.count(1))
    _orders: list[Order] = field(default_factory=list)
    _portfolio: Portfolio | None = None
    # bar index the engine is currently processing (-1 before run() starts);
    # orders remember the bar they were submitted on for TIF expiry
    _bar_index: int = -1
    _submit_bar: dict[int, int] = field(default_factory=dict)

    # -----------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------

    @property
    def portfolio(self) -> Portfolio:
        if self._portfolio is None:
            self._portfolio = Portfolio(initial_cash=self.initial_cash)
        return self._portfolio

    def submit_order(
        self,
        symbol: str,
        side: Side,
        quantity: float,
        order_type: OrderType = OrderType.MARKET,
        limit_price: float | None = None,
        stop_price: float | None = None,
        tif: TimeInForce = TimeInForce.DAY,
        client_tag: str | None = None,
    ) -> Order:
        order = Order(
            symbol=symbol,
            side=side,
            quantity=quantity,
            order_type=order_type,
            limit_price=limit_price,
            stop_price=stop_price,
            time_in_force=tif,
            client_tag=client_tag,
        )
        order.order_id = next(self._next_order_id)
        order.status = OrderStatus.WORKING
        self._orders.append(order)
        self._submit_bar[order.order_id] = self._bar_index
        return order

    def cancel_all(self, symbol: str | None = None) -> int:
        """Cancel all active orders, optionally filtered by symbol. Returns count cancelled."""
        n = 0
        for o in self._orders:
            if o.status.is_active and (symbol is None or o.symbol == symbol):
                o.cancel()
                n += 1
        return n

    def run(
        self,
        df: pd.DataFrame,
        strategy: StrategyProtocol,
    ) -> EventEngineResult:
        """Run the engine over ``df`` (must have open/high/low/close).

        On each bar:
            1. Match any working orders against the current bar's prices.
            2. Cancel unfilled IOC/FOK orders (single matching attempt).
            3. Mark-to-market the portfolio at the bar's close.
            4. Call ``strategy.on_bar(ctx)`` to allow new orders.
               (New orders are matched on the *next* bar to avoid look-ahead.)
            5. Cancel unfilled DAY orders whose session (one bar) has ended.

        Args:
            df: OHLCV DataFrame indexed by date.
            strategy: Object with an ``on_bar(ctx)`` method.

        Returns:
            :class:`EventEngineResult`.
        """
        required = {"open", "high", "low", "close"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"DataFrame is missing required columns: {missing}")

        portfolio = self.portfolio
        n = len(df)
        for i in range(n):
            ts = df.index[i]
            bar = df.iloc[i]
            self._bar_index = i

            # 1. match working orders against THIS bar
            self._match_orders_on_bar(ts, bar)

            # 2. IOC/FOK expire right after their single matching attempt
            self._expire_orders(i, (TimeInForce.IOC, TimeInForce.FOK))

            # 3. mark-to-market at the close
            portfolio.mark_to_market(ts, {self.symbol: float(bar["close"])})

            # 4. let the strategy place new orders for the next bar
            ctx = Context(
                ts=ts,
                bar=bar,
                history=df.iloc[: i + 1],
                portfolio=portfolio,
                engine=self,
                symbol=self.symbol,
            )
            strategy.on_bar(ctx)

            # 5. end-of-day expiry: DAY orders that have had their one
            #    matching session (one bar = one day in this engine) and
            #    are still unfilled cancel at the bar's close. Orders
            #    submitted during step 4 live on to the next bar.
            self._expire_orders(i, (TimeInForce.DAY,))

        equity_curve = pd.Series(
            data=[eq for _, eq in portfolio.equity_history],
            index=[ts for ts, _ in portfolio.equity_history],
            name="equity",
        )
        returns = equity_curve.pct_change().fillna(0).rename("returns")
        fills = self._build_fill_log()
        return EventEngineResult(
            equity_curve=equity_curve,
            returns=returns,
            portfolio=portfolio,
            orders=list(self._orders),
            fills=fills,
        )

    # -----------------------------------------------------------------
    # Internals
    # -----------------------------------------------------------------

    def _expire_orders(self, bar_index: int, tifs: tuple[TimeInForce, ...]) -> None:
        """Cancel still-active orders of the given TIFs that have already
        had a matching opportunity (submitted before ``bar_index``)."""
        for o in self._orders:
            if not o.status.is_active or o.time_in_force not in tifs:
                continue
            if self._submit_bar.get(o.order_id or -1, -1) < bar_index:
                o.cancel()

    def _match_orders_on_bar(self, ts: pd.Timestamp, bar: pd.Series) -> None:
        """Match all working orders against the bar's prices.

        Order-of-execution per bar:
            STOPs trigger first if the bar touches the stop_price,
            then LIMITs fill if the bar's range covers limit_price,
            then MARKETs fill at the bar's open.
        """
        open_p = float(bar["open"])
        high_p = float(bar["high"])
        low_p = float(bar["low"])

        for order in self._orders:
            if not order.status.is_active:
                continue
            if order.symbol != self.symbol:
                continue

            fill_price: float | None = None

            if order.order_type is OrderType.MARKET:
                fill_price = open_p

            elif order.order_type is OrderType.LIMIT:
                limit = order.limit_price
                assert limit is not None  # guaranteed for LIMIT (Order.__post_init__)
                # buy LIMIT fills if bar's low <= limit_price (price came down to us)
                # sell LIMIT fills if bar's high >= limit_price (price came up to us)
                if order.side is Side.BUY and low_p <= limit:
                    fill_price = min(limit, open_p)  # gap protection
                elif order.side is Side.SELL and high_p >= limit:
                    fill_price = max(limit, open_p)

            elif order.order_type is OrderType.STOP:
                stop = order.stop_price
                assert stop is not None  # guaranteed for STOP (Order.__post_init__)
                # buy STOP triggers when price rises through stop_price
                # sell STOP triggers when price falls through stop_price
                triggered = (order.side is Side.BUY and high_p >= stop) or (
                    order.side is Side.SELL and low_p <= stop
                )
                if triggered:
                    # convert to market, fill at max(open, stop) for buy / min for sell
                    fill_price = max(open_p, stop) if order.side is Side.BUY else min(open_p, stop)

            elif order.order_type is OrderType.STOP_LIMIT:
                stop = order.stop_price
                limit = order.limit_price
                assert stop is not None and limit is not None  # guaranteed for STOP_LIMIT
                triggered = (order.side is Side.BUY and high_p >= stop) or (
                    order.side is Side.SELL and low_p <= stop
                )
                # behave as LIMIT post-trigger, with the same gap
                # protection as plain LIMITs: a bar that opens through
                # the stop makes the order live from the open, so it
                # fills at the better of open/limit; an intrabar trigger
                # fills at the limit (conservative).
                if triggered and (
                    (order.side is Side.BUY and low_p <= limit)
                    or (order.side is Side.SELL and high_p >= limit)
                ):
                    if order.side is Side.BUY:
                        fill_price = min(limit, open_p) if open_p >= stop else limit
                    else:
                        fill_price = max(limit, open_p) if open_p <= stop else limit

            if fill_price is None:
                continue

            # apply adverse slippage
            slip = self.slippage_bps / 10_000.0
            if order.side is Side.BUY:
                fill_price *= 1 + slip
            else:
                fill_price *= 1 - slip

            fill_qty = order.remaining_quantity
            commission = (
                max(
                    fill_qty * self.commission_per_share,
                    self.commission_min,
                )
                if (self.commission_per_share > 0 or self.commission_min > 0)
                else 0.0
            )

            # LIMIT orders rest in the book → they earn the maker side;
            # MARKET / STOP / STOP_LIMIT cross the spread → taker.
            liquidity = Liquidity.MAKER if order.order_type is OrderType.LIMIT else Liquidity.TAKER
            order.record_fill(fill_qty, fill_price, when=ts, liquidity=liquidity)
            self.portfolio.record_fill(
                symbol=order.symbol,
                side=order.side,
                quantity=fill_qty,
                price=fill_price,
                commission=commission,
            )

    def _build_fill_log(self) -> pd.DataFrame:
        rows: list[dict] = []
        for o in self._orders:
            for f in o.fills:
                rows.append(
                    {
                        "ts": f.ts,
                        "seq": f.seq,
                        "order_id": o.order_id,
                        "symbol": o.symbol,
                        "side": o.side.value,
                        "quantity": f.quantity,
                        "price": f.price,
                        "liquidity": f.liquidity.value,
                        "tag": o.client_tag,
                    }
                )
        return pd.DataFrame(rows)
