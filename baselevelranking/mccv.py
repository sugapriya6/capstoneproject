# ===============================
# MCCV‑BASED CLASSIFIER RANKING
# ===============================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# -------------------------------------------------
# 1. LOAD MCCV RESULTS (CSV FILE)
# -------------------------------------------------
# CSV format:
# Classifier,Metric,Mean,Std

csv_file = "C:/capstone/alzheimers/proposed/mccv/All_Classifiers_Mean_Std_MCCV.csv"
df = pd.read_csv(csv_file)

# -------------------------------------------------
# 2. DEFINE CLASSIFIERS (FIXED ORDER)
# -------------------------------------------------
classifiers = ["RF", "LR", "LDA", "GNB", "ET", "XGB", "KNN", "SVM", "MLP", "DT"]

# -------------------------------------------------
# 3. CONVERT CSV TO METRIC DICTIONARY
# -------------------------------------------------
metrics = {}

for metric in df["Metric"].unique():
    sub = df[df["Metric"] == metric]

    metrics[metric] = {
        "mean": [sub[sub["Classifier"] == c]["Mean"].values[0] for c in classifiers],
        "std":  [sub[sub["Classifier"] == c]["Std"].values[0]  for c in classifiers]
    }

# -------------------------------------------------
# 4. MCCV RANKING PLOT FUNCTION
# -------------------------------------------------
def plot_mccv(classifiers, metrics, title, save_name):
    fig, axes = plt.subplots(2, 5, figsize=(18, 8))
    fig.suptitle(title, fontsize=16)

    axes = axes.flatten()

    for i, (metric, data) in enumerate(metrics.items()):
        mean_vals = data["mean"]
        std_vals = data["std"]

        # Stability ratio (Std / Mean)
        ratio = [(s / m) if m != 0 else 0 for s, m in zip(std_vals, mean_vals)]

        # Sort classifiers (lower ratio = better)
        sorted_cls, sorted_ratio = zip(
            *sorted(zip(classifiers, ratio), key=lambda x: x[1])
        )

        axes[i].bar(sorted_cls, sorted_ratio)
        axes[i].set_title(metric)
        axes[i].set_ylabel("Std / Mean")
        axes[i].set_xticklabels(sorted_cls, rotation=45)

    plt.tight_layout()
    plt.subplots_adjust(top=0.88)
    plt.savefig(save_name, dpi=300)
    plt.show()

# -------------------------------------------------
# 5. GENERATE MCCV RANKING PLOT
# -------------------------------------------------
plot_mccv(
    classifiers,
    metrics,
    "MCCV‑Based Ranking of Classifiers",
    "MCCV_Ranking.png"
)

# -------------------------------------------------
# 6. SAVE MCCV RANKING TABLE
# -------------------------------------------------
rows = []

for metric, data in metrics.items():
    for c, m, s in zip(classifiers, data["mean"], data["std"]):
        rows.append([c, metric, m, s, (s / m) if m != 0 else 0])

ranking_df = pd.DataFrame(
    rows,
    columns=["Classifier", "Metric", "Mean", "Std", "Std_Mean_Ratio"]
)

ranking_df.to_csv("MCCV_Ranking_Table.csv", index=False)

print("✅ MCCV ranking plot saved as MCCV_Ranking.png")
print("✅ MCCV ranking table saved as MCCV_Ranking_Table.csv")
