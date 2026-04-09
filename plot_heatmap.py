import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# wczytaj wyniki grid search
df = pd.read_csv("results/grid_search_results.csv")

# pivot do heatmapy
heatmap_data = df.pivot(
    index="lookback",
    columns="threshold",
    values="sharpe"
)

plt.figure(figsize=(8, 6))

sns.heatmap(
    heatmap_data,
    annot=True,
    cmap="RdYlGn",
    fmt=".2f"
)

plt.title("Momentum Strategy Sharpe Ratio Heatmap")
plt.xlabel("Threshold")
plt.ylabel("Lookback")

plt.tight_layout()
plt.savefig("results/parameter_heatmap.png")

plt.show()