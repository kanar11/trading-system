"""Corporate-action bookkeeping on a live Portfolio.

:mod:`src.data.corporate_actions` back-adjusts *price series* for
research; a live book needs the other side: when a split or dividend
actually happens, the **positions and cash** must be restated or the
portfolio's equity and P&L go wrong on the ex-date. These helpers apply
the events to a :class:`~src.oms.portfolio.Portfolio` in place (they are
bookkeeping, like ``record_fill`` — not pure planning):

* a split multiplies the position quantity by the ratio and divides the
  average price by it — cost basis, realised and unrealised P&L are all
  invariant;
* a cash dividend credits ``quantity × amount`` to cash — longs receive
  it, shorts *pay* it (payment in lieu), exactly as the signed quantity
  implies.

``total_traded_qty`` is deliberately left in pre-split units (it is a
historical turnover record, not a position).
"""

from __future__ import annotations

import math

from src.oms.portfolio import Portfolio


def apply_split(portfolio: Portfolio, symbol: str, ratio: float) -> float:
    """Restate a position for a stock split (in place).

    Args:
        portfolio: The book to restate (mutated).
        symbol: The split symbol; holding nothing in it is a no-op.
        ratio: Shares-out per share-in (2.0 = 2-for-1 split, 0.25 =
            1-for-4 reverse split).

    Returns:
        The position's new signed quantity (0.0 when nothing was held).

    Raises:
        ValueError: If ``ratio`` is not a positive finite number.
    """
    if not math.isfinite(ratio) or ratio <= 0:
        raise ValueError(f"ratio must be finite and > 0, got {ratio}.")

    position = portfolio.positions.get(symbol)
    if position is None or position.is_flat:
        return 0.0
    position.quantity *= ratio
    position.avg_price /= ratio  # cost basis quantity*avg_price is invariant
    return position.quantity


def apply_dividend(portfolio: Portfolio, symbol: str, amount_per_share: float) -> float:
    """Book a cash dividend against the held position (in place).

    Longs are credited ``quantity × amount``; shorts are debited the same
    (payment in lieu). The equity jump on the ex-date is the model's
    counterpart of the price drop the market applies.

    Args:
        portfolio: The book to credit/debit (mutated).
        symbol: The dividend-paying symbol; holding nothing is a no-op.
        amount_per_share: Cash amount per share (>= 0).

    Returns:
        The signed cash flow applied (0.0 when nothing was held).

    Raises:
        ValueError: If ``amount_per_share`` is negative or not finite.
    """
    if not math.isfinite(amount_per_share) or amount_per_share < 0:
        raise ValueError(f"amount_per_share must be finite and >= 0, got {amount_per_share}.")

    position = portfolio.positions.get(symbol)
    if position is None or position.is_flat:
        return 0.0
    cash_flow = position.quantity * amount_per_share
    portfolio.cash += cash_flow
    return cash_flow
