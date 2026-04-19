"""Plotting utilities for backtest results."""

import logging

import matplotlib.pyplot as plt
import pandas as pd

logger = logging.getLogger(__name__)


def plot_equity(
    result: pd.DataFrame,
    title: str = "Equity Curve",
    save_path: str | None = None,
    show: bool = True,
) -> None:
    """Plot the equity curve from a backtest result.

    Args:
        result: DataFrame containing an 'equity_curve' column.
        title: Chart title.
        save_path: If provided, save the figure to this path.
        show: Whether to display the plot interactively.

    Raises:
        ValueError: If 'equity_curve' column is missing.
    """
    if "equity_curve" not in result.columns:
        raise ValueError("'equity_curve' column missing from DataFrame.")

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(result.index, result["equity_curve"], linewidth=1.2)
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Equity")
    ax.grid(alpha=0.3)
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150)

    if show:
        plt.show()

    plt.close(fig)
