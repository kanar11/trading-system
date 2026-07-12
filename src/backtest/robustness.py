"""Backtest robustness checks: execution-lag sensitivity.

A momentum result that only exists if you trade the instant the signal
prints is not a result — it is a microstructure artefact. The standard
industry robustness check is to re-run the backtest with the signal
delayed by 1, 2, ... extra bars and watch how gracefully performance
degrades: genuine slow-moving edges (12-1 momentum, trend) survive a few
days of delay, while look-ahead bugs and fragile timing effects collapse
at lag 1.

:func:`lag_sensitivity` automates that table on top of the vectorised
:func:`src.backtest.engine.backtest_strategy` (which already executes
next-bar; ``lag`` is *additional* delay on top of that convention).
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from src.backtest.engine import backtest_strategy


def lag_sensitivity(
    df: pd.DataFrame,
    lags: Sequence[int] = (0, 1, 2, 3, 4, 5),
    transaction_cost: float = 0.001,
    periods_per_year: int = 252,
) -> pd.DataFrame:
    """Re-run a signal backtest with extra execution delay per lag.

    Args:
        df: DataFrame with ``close`` and ``signal`` columns (the input of
            :func:`~src.backtest.engine.backtest_strategy`).
        lags: Extra delays in bars to test; 0 is the engine's baseline
            next-bar execution.
        transaction_cost: Round-trip cost fraction passed to the engine.
        periods_per_year: Bars per year for annualisation.

    Returns:
        DataFrame indexed by lag with columns ``ann_return``, ``ann_vol``,
        ``sharpe`` (NaN when the volatility is zero), ``max_drawdown``
        (<= 0) and ``final_equity``.

    Raises:
        ValueError: If ``lags`` is empty or contains a negative value
            (missing columns raise from the engine).
    """
    if len(lags) == 0:
        raise ValueError("lags must not be empty.")
    if any(lag < 0 for lag in lags):
        raise ValueError(f"lags must be >= 0, got {list(lags)}.")
    if periods_per_year < 1:
        raise ValueError(f"periods_per_year must be >= 1, got {periods_per_year}.")

    rows = []
    for lag in lags:
        delayed = df.copy()
        delayed["signal"] = delayed["signal"].shift(lag).fillna(0)
        result, _ = backtest_strategy(delayed, transaction_cost=transaction_cost)

        returns = result["strategy_returns"]
        mean = float(returns.mean())
        std = float(returns.std(ddof=1))
        ann_return = mean * periods_per_year
        ann_vol = std * float(np.sqrt(periods_per_year))
        equity = result["equity_curve"]
        drawdown = equity / equity.cummax() - 1.0

        rows.append(
            {
                "lag": int(lag),
                "ann_return": ann_return,
                "ann_vol": ann_vol,
                "sharpe": ann_return / ann_vol if ann_vol > 0 else float("nan"),
                "max_drawdown": float(drawdown.min()),
                "final_equity": float(equity.iloc[-1]),
            }
        )
    return pd.DataFrame(rows).set_index("lag")
