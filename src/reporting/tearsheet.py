"""Multi-panel tear-sheet report.

Produces a single PNG (or matplotlib Figure) that summarises a backtest:

    [ 1 ] Equity curve vs benchmark
    [ 2 ] Underwater drawdown
    [ 3 ] Rolling Sharpe (default 63-day)
    [ 4 ] Monthly returns heatmap (year x month)
    [ 5 ] Returns distribution histogram with vol bands
    [ 6 ] Performance metrics table

All panels are optional — pass ``benchmark=None`` to drop the
benchmark trace, ``trade_log=None`` to drop the trade-stats line.

Headless friendly: uses the Agg backend implicitly via matplotlib's
``Figure`` constructor (no ``plt.show``). Safe to call from CI.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from src.reporting.metrics import calculate_metrics, calculate_trade_stats

logger = logging.getLogger(__name__)


def _drawdown_series(returns: pd.Series) -> pd.Series:
    equity = (1 + returns.fillna(0)).cumprod()
    return equity / equity.cummax() - 1


def _rolling_sharpe(returns: pd.Series, window: int = 63) -> pd.Series:
    mean = returns.rolling(window).mean()
    std = returns.rolling(window).std()
    return (mean / std.replace(0, np.nan)) * np.sqrt(252)


def _monthly_return_matrix(returns: pd.Series) -> pd.DataFrame:
    """Pivot daily returns into a year (rows) x month (cols) matrix."""
    monthly = (1 + returns.fillna(0)).resample("ME").prod() - 1
    df = monthly.to_frame("ret")
    df["year"] = df.index.year
    df["month"] = df.index.month
    return df.pivot(index="year", columns="month", values="ret")


def generate_tearsheet(
    returns: pd.Series,
    output_path: str | Path | None = None,
    benchmark: pd.Series | None = None,
    trade_log: pd.DataFrame | None = None,
    title: str = "Strategy Tearsheet",
    rolling_window: int = 63,
) -> Figure:
    """Build a multi-panel tear-sheet and optionally save it as PNG.

    Args:
        returns: Daily strategy returns (not cumulative).
        output_path: If given, save the figure to this path as PNG.
        benchmark: Optional benchmark daily-return series (aligned by date).
        trade_log: Optional trade log; powers the bottom-row trade stats.
        title: Figure title.
        rolling_window: Window for the rolling-Sharpe panel.

    Returns:
        The matplotlib Figure (caller may further annotate / close it).
    """
    returns = returns.dropna()
    if returns.empty:
        raise ValueError("returns is empty - nothing to plot")

    metrics = calculate_metrics(returns)
    trade_stats = calculate_trade_stats(trade_log) if trade_log is not None else None

    fig = plt.figure(figsize=(14, 12))
    gs = fig.add_gridspec(4, 2, hspace=0.45, wspace=0.25)

    # [1] Equity curve
    ax_eq = fig.add_subplot(gs[0, :])
    equity = (1 + returns).cumprod()
    ax_eq.plot(equity.index, equity.values, label="Strategy", linewidth=1.3)
    if benchmark is not None:
        bench = benchmark.reindex(returns.index).fillna(0)
        bench_eq = (1 + bench).cumprod()
        ax_eq.plot(bench_eq.index, bench_eq.values, label="Benchmark", linewidth=1.0, alpha=0.7)
    ax_eq.set_title("Equity curve")
    ax_eq.set_ylabel("Equity")
    ax_eq.legend(loc="upper left")
    ax_eq.grid(alpha=0.3)

    # [2] Underwater drawdown
    ax_dd = fig.add_subplot(gs[1, 0])
    dd = _drawdown_series(returns)
    ax_dd.fill_between(dd.index, dd.values, 0, color="tab:red", alpha=0.4)
    ax_dd.set_title("Underwater drawdown")
    ax_dd.set_ylabel("Drawdown")
    ax_dd.grid(alpha=0.3)

    # [3] Rolling Sharpe
    ax_rs = fig.add_subplot(gs[1, 1])
    rs = _rolling_sharpe(returns, window=rolling_window)
    ax_rs.plot(rs.index, rs.values, color="tab:blue", linewidth=1.0)
    ax_rs.axhline(0, color="black", linewidth=0.5)
    ax_rs.set_title(f"Rolling Sharpe ({rolling_window}d)")
    ax_rs.set_ylabel("Sharpe")
    ax_rs.grid(alpha=0.3)

    # [4] Monthly returns heatmap
    ax_hm = fig.add_subplot(gs[2, :])
    mat = _monthly_return_matrix(returns)
    if not mat.empty:
        # symmetric colour scale around 0
        vmax = float(np.nanmax(np.abs(mat.values))) if mat.size > 0 else 0.0
        vmax = vmax if vmax > 0 else 1.0
        im = ax_hm.imshow(
            mat.values,
            aspect="auto",
            cmap="RdYlGn",
            vmin=-vmax,
            vmax=vmax,
        )
        ax_hm.set_xticks(range(mat.shape[1]))
        ax_hm.set_xticklabels(
            ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"][
                : mat.shape[1]
            ]
        )
        ax_hm.set_yticks(range(mat.shape[0]))
        ax_hm.set_yticklabels(mat.index)
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                v = mat.values[i, j]
                if not np.isnan(v):
                    ax_hm.text(
                        j,
                        i,
                        f"{v:.1%}",
                        ha="center",
                        va="center",
                        fontsize=7,
                        color="black" if abs(v) < vmax * 0.5 else "white",
                    )
        fig.colorbar(im, ax=ax_hm, shrink=0.7, format="%.0%%")
    ax_hm.set_title("Monthly returns")

    # [5] Returns distribution
    ax_hist = fig.add_subplot(gs[3, 0])
    ax_hist.hist(returns.values, bins=50, color="tab:blue", alpha=0.7)
    sigma = returns.std()
    ax_hist.axvline(0, color="black", linewidth=0.5)
    ax_hist.axvline(sigma, color="grey", linewidth=0.5, linestyle="--", label="±1σ")
    ax_hist.axvline(-sigma, color="grey", linewidth=0.5, linestyle="--")
    ax_hist.set_title("Daily returns distribution")
    ax_hist.set_xlabel("Return")
    ax_hist.legend(fontsize=8)
    ax_hist.grid(alpha=0.3)

    # [6] Metrics table
    ax_tbl = fig.add_subplot(gs[3, 1])
    ax_tbl.axis("off")
    rows = [
        ("Total Return", f"{metrics['Total Return']:.1%}"),
        ("CAGR", f"{metrics['CAGR']:.1%}"),
        ("Sharpe", f"{metrics['Sharpe Ratio']:.2f}"),
        ("Sortino", f"{metrics['Sortino Ratio']:.2f}"),
        ("Max DD", f"{metrics['Max Drawdown']:.1%}"),
        ("Calmar", f"{metrics['Calmar Ratio']:.2f}"),
    ]
    if trade_stats and trade_stats["Total Trades"] > 0:
        rows.extend(
            [
                ("Trades", f"{trade_stats['Total Trades']}"),
                ("Win Rate", f"{trade_stats['Win Rate']:.1%}"),
                ("Profit Factor", f"{trade_stats['Profit Factor']:.2f}"),
            ]
        )
    table = ax_tbl.table(
        cellText=rows,
        colLabels=["Metric", "Value"],
        loc="center",
        cellLoc="left",
        colWidths=[0.5, 0.5],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.4)
    ax_tbl.set_title("Performance")

    fig.suptitle(title, fontsize=14, fontweight="bold", y=0.995)

    if output_path is not None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=130, bbox_inches="tight")
        logger.info("Saved tear-sheet to %s", path)

    return fig
