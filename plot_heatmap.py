"""Generate a Sharpe Ratio heatmap from sweep or grid search results.

Usage:
    python plot_heatmap.py
    python plot_heatmap.py --csv results/grid_search_results.csv
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot_heatmap(csv_path: str = "results/sweep_results.csv") -> None:
    """Read sweep/grid search results and create a lookback x threshold heatmap.

    Automatically detects the Sharpe column name ('sharpe' or 'Sharpe Ratio').

    Args:
        csv_path: Path to the CSV file with sweep results.
    """
    path = Path(csv_path)
    if not path.exists():
        # fallback to grid_search output if sweep_results not found
        fallback = Path("results/grid_search_results.csv")
        if fallback.exists():
            path = fallback
        else:
            raise FileNotFoundError(
                f"Neither '{csv_path}' nor '{fallback}' found. "
                "Run grid_search.py or sweep.py first."
            )

    df = pd.read_csv(path)

    # detect sharpe column name
    sharpe_col = "sharpe"
    if sharpe_col not in df.columns and "Sharpe Ratio" in df.columns:
        sharpe_col = "Sharpe Ratio"

    pivot = df.pivot_table(
        index="lookback",
        columns="threshold",
        values=sharpe_col,
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
    parser = argparse.ArgumentParser(description="Plot Sharpe Ratio heatmap.")
    parser.add_argument(
        "--csv",
        default="results/sweep_results.csv",
        help="Path to sweep/grid search CSV (default: results/sweep_results.csv)",
    )
    args = parser.parse_args()
    plot_heatmap(args.csv)
