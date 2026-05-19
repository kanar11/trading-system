"""Multi-asset portfolio backtest.

Runs the same single-instrument strategy / backtest pipeline across
several tickers, then aggregates the per-asset return streams into a
portfolio return series using one of three weighting schemes:

    * ``equal``         — flat 1/N weights, rebalanced daily.
    * ``inverse_vol``   — weights inversely proportional to trailing
                          realised volatility (risk-parity).
    * ``custom``        — user-supplied static weights.

This is intentionally a "research portfolio" — full rebalancing
costs, correlation-aware risk budgeting and capital constraints are
out of scope. The goal is to see whether a single-instrument edge
survives diversification across a basket.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, Mapping

import numpy as np
import pandas as pd

from src.reporting.metrics import calculate_metrics

logger = logging.getLogger(__name__)


WeightingScheme = str  # "equal" | "inverse_vol" | "custom"


@dataclass
class PortfolioConfig:
    """Configuration for a portfolio-level backtest.

    Attributes:
        weighting: One of "equal", "inverse_vol", "custom".
        vol_window: Rolling window (days) for the inverse-vol estimate.
        custom_weights: Mapping of ticker → weight when weighting="custom".
            Weights are normalised to sum to 1 across the supplied tickers.
        rebalance_freq: How often to rebalance — "D" daily, "W" weekly,
            "M" monthly. Daily is the default.
    """

    weighting: WeightingScheme = "equal"
    vol_window: int = 20
    custom_weights: Mapping[str, float] | None = None
    rebalance_freq: str = "D"


@dataclass
class PortfolioResult:
    """Aggregated result of a portfolio backtest.

    Attributes:
        returns: Per-ticker daily return frame (columns = tickers).
        weights: Per-ticker daily weight frame (columns = tickers).
        portfolio_returns: Weighted portfolio daily returns.
        equity_curve: Cumulative portfolio equity (starting at 1.0).
        metrics: Portfolio-level performance metrics.
        per_asset_metrics: Per-ticker performance metrics.
    """

    returns: pd.DataFrame
    weights: pd.DataFrame
    portfolio_returns: pd.Series
    equity_curve: pd.Series
    metrics: dict[str, float]
    per_asset_metrics: dict[str, dict[str, float]] = field(default_factory=dict)


def _compute_weights(
    returns: pd.DataFrame,
    config: PortfolioConfig,
) -> pd.DataFrame:
    """Build the weights matrix for the supplied per-asset return frame."""
    tickers = list(returns.columns)
    n = len(tickers)

    if config.weighting == "equal":
        weights = pd.DataFrame(
            np.full((len(returns), n), 1.0 / n),
            index=returns.index,
            columns=tickers,
        )

    elif config.weighting == "inverse_vol":
        rolling_vol = returns.rolling(config.vol_window).std()
        inv = 1.0 / rolling_vol.replace(0, np.nan)
        # row-normalise; if every entry is NaN (warm-up), fall back to equal
        row_sum = inv.sum(axis=1)
        weights = inv.div(row_sum, axis=0)
        warmup_mask = row_sum.isna() | (row_sum == 0)
        weights.loc[warmup_mask] = 1.0 / n
        weights = weights.fillna(0.0)

    elif config.weighting == "custom":
        if not config.custom_weights:
            raise ValueError("custom_weights required when weighting='custom'")
        raw = np.array(
            [config.custom_weights.get(t, 0.0) for t in tickers], dtype=float
        )
        total = raw.sum()
        if total <= 0:
            raise ValueError("custom_weights must sum to a positive value")
        raw = raw / total
        weights = pd.DataFrame(
            np.broadcast_to(raw, (len(returns), n)),
            index=returns.index,
            columns=tickers,
        ).copy()

    else:
        raise ValueError(f"unknown weighting scheme: {config.weighting!r}")

    # rebalance frequency — hold weights constant between rebalance dates
    if config.rebalance_freq.upper() != "D":
        resampled = weights.resample(config.rebalance_freq.upper()).first()
        weights = resampled.reindex(weights.index, method="ffill").fillna(0.0)

    return weights


def run_portfolio_backtest(
    data: Mapping[str, pd.DataFrame],
    strategy_fn: Callable[[pd.DataFrame], pd.DataFrame],
    backtest_fn: Callable[[pd.DataFrame], tuple[pd.DataFrame, pd.DataFrame]],
    config: PortfolioConfig | None = None,
) -> PortfolioResult:
    """Run a strategy across multiple tickers and aggregate into a portfolio.

    Args:
        data: Mapping of ticker → OHLCV DataFrame. Each frame must have
            a 'close' column and share an aligned date index (will be
            inner-joined automatically).
        strategy_fn: Single-asset strategy function (df → df_with_signal).
        backtest_fn: Single-asset backtest function (df → (bt, trades)).
        config: Portfolio aggregation config.

    Returns:
        A :class:`PortfolioResult` with per-asset and combined output.
    """
    if not data:
        raise ValueError("data must contain at least one ticker")
    config = config or PortfolioConfig()

    per_asset_returns: dict[str, pd.Series] = {}
    per_asset_metrics: dict[str, dict[str, float]] = {}

    for ticker, df in data.items():
        logger.info("Portfolio leg: running backtest on %s", ticker)
        sig_df = strategy_fn(df)
        bt_df, _ = backtest_fn(sig_df)
        per_asset_returns[ticker] = bt_df["strategy_returns"].rename(ticker)
        per_asset_metrics[ticker] = calculate_metrics(bt_df["strategy_returns"])

    # align on the common date range (inner join)
    returns_df = pd.concat(per_asset_returns.values(), axis=1, join="inner").fillna(0.0)

    weights = _compute_weights(returns_df, config)
    # safety: align weights to the return matrix in case of resample drift
    weights = weights.reindex(returns_df.index).fillna(0.0)

    portfolio_returns = (returns_df * weights).sum(axis=1)
    equity_curve = (1 + portfolio_returns).cumprod()
    metrics = calculate_metrics(portfolio_returns)
    metrics["N Assets"] = len(returns_df.columns)

    return PortfolioResult(
        returns=returns_df,
        weights=weights,
        portfolio_returns=portfolio_returns,
        equity_curve=equity_curve,
        metrics=metrics,
        per_asset_metrics=per_asset_metrics,
    )
