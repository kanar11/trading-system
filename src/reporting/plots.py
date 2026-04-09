import matplotlib.pyplot as plt
import pandas as pd

def plot_equity(result: pd.DataFrame, title: str = "Equity Curve", save_path: str | None = None):
    if "equity_curve" not in result.columns:
        raise ValueError("'equity_curve' column missing")

    ax = result["equity_curve"].plot(title=title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Equity")
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=150)

    plt.show()