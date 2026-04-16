"""Generate a Sharpe Ratio heatmap from sweep results."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot_heatmap(csv_path: str = "results/sweep_results.csv") -> None:
    """Read sweep results and create a lookback x threshold heatmap.

    Args:
        csv_path: Path to the sweep CSV file.
    """
    df = pd.read_csv(csv_path)

    pivot = df.pivot_table(
        index="lookback",
        columns="threshold",
        values="sharpe",
        aggfunc="first",
    )

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(pivot, annot=True, fmt=".2f", cmap="RdYlGn", ax=ax)
    ax.set_title("Sharpe Ratio — Lookback x Threshold")
    ax.set_ylabel("Lookback")
    ax.set_xlabel("Threshold")
    fig.tight_layout()

    output_path = Path("results") / "parameter_heatmap.png"
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    plot_heatmap()
