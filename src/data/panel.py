"""Build wide price panels from per-ticker OHLCV frames.

The multi-asset pipeline (``dual_momentum`` → ``backtest_weights`` →
``rebalance_orders``) consumes a *wide* close frame, but every loader in
:mod:`src.data` produces one OHLCV frame per ticker — and stitching them
by hand keeps reinventing the same calendar-alignment decisions. This
helper pins them down: pick a column, align the tickers on a common
calendar (``inner`` join — only shared timestamps, guaranteed NaN-free
when the inputs are, which is exactly what ``backtest_weights``
requires), or keep the union (``outer``) with optional bounded forward
fill for tickers that miss the odd session.

Direct-import module::

    from src.data.panel import build_close_frame
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Literal

import pandas as pd


def build_close_frame(
    frames: Mapping[str, pd.DataFrame],
    column: str = "close",
    join: str = "inner",
    ffill_limit: int | None = None,
) -> pd.DataFrame:
    """Assemble a wide one-column-per-ticker panel from OHLCV frames.

    Args:
        frames: ``{ticker: OHLCV frame}``; each frame needs ``column`` and
            a sorted, unique DatetimeIndex.
        column: Which column to extract (``"close"``, ``"adj_close"``...).
        join: ``"inner"`` keeps only timestamps present for every ticker;
            ``"outer"`` keeps the union (missing sessions become NaN).
        ffill_limit: When set, forward-fill gaps up to this many bars
            after joining (leading NaNs are never filled).

    Returns:
        Wide DataFrame, columns in ``frames`` insertion order.

    Raises:
        TypeError: If an index is not a DatetimeIndex.
        ValueError: If ``frames`` is empty, a column is missing, an index
            is unsorted/duplicated, ``join`` is unknown, ``ffill_limit``
            < 1, or an inner join leaves no common timestamps.
    """
    if not frames:
        raise ValueError("frames must not be empty.")
    if join not in ("inner", "outer"):
        raise ValueError(f"join must be 'inner' or 'outer', got {join!r}.")
    if ffill_limit is not None and ffill_limit < 1:
        raise ValueError(f"ffill_limit must be >= 1 when set, got {ffill_limit}.")

    series = []
    for ticker, frame in frames.items():
        if column not in frame.columns:
            raise ValueError(f"frame for {ticker!r} is missing column {column!r}.")
        index = frame.index
        if not isinstance(index, pd.DatetimeIndex):
            raise TypeError(f"frame for {ticker!r} must have a DatetimeIndex.")
        if not index.is_monotonic_increasing:
            raise ValueError(f"frame for {ticker!r} has an unsorted index.")
        if index.has_duplicates:
            raise ValueError(f"frame for {ticker!r} has duplicate timestamps.")
        series.append(frame[column].rename(ticker))

    join_literal: Literal["inner", "outer"] = "inner" if join == "inner" else "outer"
    panel = pd.concat(series, axis=1, join=join_literal)
    if join == "inner" and panel.empty:
        raise ValueError("inner join produced no common timestamps.")
    if ffill_limit is not None:
        panel = panel.ffill(limit=ffill_limit)
    return panel
