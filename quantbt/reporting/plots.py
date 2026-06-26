"""Plotting utilities for backtest results."""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.figure import Figure

logger = logging.getLogger(__name__)


def plot_equity(
    result: pd.DataFrame,
    title: str = "Equity Curve",
    benchmark: pd.Series | None = None,
    benchmark_label: str = "Benchmark",
    save_path: str | Path | None = None,
    show: bool = False,
) -> Figure:
    """Plot the equity curve from a backtest result.

    Args:
        result: DataFrame containing an 'equity_curve' column.
        title: Chart title.
        benchmark: Optional benchmark equity series to overlay (e.g. buy-and-hold).
        benchmark_label: Legend label for the benchmark line.
        save_path: If provided, save the figure to this path. Parent
            directories are created automatically.
        show: Whether to display the plot interactively.

    Returns:
        The created matplotlib Figure (already closed if ``show`` is False
        and a ``save_path`` was given is *not* assumed — the caller owns it).

    Raises:
        ValueError: If the 'equity_curve' column is missing.
    """
    if "equity_curve" not in result.columns:
        raise ValueError("'equity_curve' column missing from DataFrame.")

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(result.index, result["equity_curve"], linewidth=1.2, label="Strategy")

    if benchmark is not None:
        ax.plot(benchmark.index, benchmark, linewidth=1.2, label=benchmark_label)
        ax.legend()

    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Equity")
    ax.grid(alpha=0.3)
    fig.tight_layout()

    if save_path is not None:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150)
        logger.info("Saved equity plot: %s", path)

    if show:
        plt.show()

    return fig
