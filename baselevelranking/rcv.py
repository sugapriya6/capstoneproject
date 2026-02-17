import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# ===============================
# Output directory
# ===============================
OUTPUT_DIR = "rcv_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ===============================
# Function to plot & save RCV rankings
# ===============================
def plot_metrics_and_save(classifiers, metrics, title):
    fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(20, 9))
    fig.suptitle(title, fontsize=16)
    axes = axes.flatten()

    ranking_table = []

    for i, (metric, data) in enumerate(metrics.items()):
        ax = axes[i]

        mean_values = data["mean"]
        std_values = data["std"]

        # Base-level ranking formula
        ratio = [(std / mean) if mean != 0 else 0
                 for mean, std in zip(mean_values, std_values)]

        # Store for CSV
        for cls, r in zip(classifiers, ratio):
            ranking_table.append({
                "Metric": metric,
                "Classifier": cls,
                "Std_to_Mean_Ratio": r
            })

        # Sort for plotting
        sorted_cls, sorted_ratio = zip(
            *sorted(zip(classifiers, ratio), key=lambda x: x[1])
        )

        ax.bar(sorted_cls, sorted_ratio, color="steelblue")
        ax.set_title(metric)
        ax.set_ylabel("Std / Mean")
        ax.set_ylim(0, max(sorted_ratio) * 1.2)
        ax.set_xticklabels(sorted_cls, rotation=45, ha="right")

    plt.tight_layout()
    plt.subplots_adjust(top=0.90)

    # ===============================
    # Save plot
    # ===============================
    plot_path = os.path.join(OUTPUT_DIR, "RCV_Base_Level_Ranking.png")
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.show()

    # ===============================
    # Save CSV
    # ===============================
    df_rank = pd.DataFrame(ranking_table)
    csv_path = os.path.join(OUTPUT_DIR, "RCV_Base_Level_Ranking.csv")
    df_rank.to_csv(csv_path, index=False)

    print(f"✔ Plot saved to: {plot_path}")
    print(f"✔ Ranking table saved to: {csv_path}")


# ===============================
# Classifier order (fixed)
# ===============================
classifiers = ["RF", "LR", "LDA", "GNB", "ET", "XGB", "KNN", "SVM", "MLP", "DT"]


# ===============================
# RCV Metrics (YOUR VALUES)
# ===============================
rcv_metrics = {
    "Accuracy": {
        "mean": [0.800,0.786,0.765,0.766,0.809,0.802,0.732,0.770,0.760,0.695],
        "std":  [0.092,0.094,0.108,0.093,0.090,0.090,0.093,0.097,0.093,0.108]
    },
    "Precision": {
        "mean": [0.799,0.815,0.800,0.789,0.819,0.803,0.875,0.790,0.777,0.708],
        "std":  [0.106,0.110,0.125,0.112,0.104,0.104,0.129,0.112,0.105,0.119]
    },
    "Sensitivity (Recall)": {
        "mean": [0.831,0.766,0.734,0.761,0.821,0.828,0.566,0.768,0.761,0.706],
        "std":  [0.132,0.141,0.157,0.131,0.133,0.116,0.152,0.133,0.150,0.161]
    },
    "Specificity": {
        "mean": [0.768,0.807,0.797,0.771,0.796,0.774,0.906,0.772,0.759,0.685],
        "std":  [0.137,0.126,0.139,0.142,0.137,0.138,0.099,0.139,0.129,0.149]
    },
    "F1 Score": {
        "mean": [0.808,0.782,0.756,0.766,0.812,0.810,0.675,0.771,0.759,0.698],
        "std":  [0.093,0.102,0.119,0.095,0.091,0.087,0.131,0.100,0.104,0.120]
    },
    "ROC AUC": {
        "mean": [0.886,0.880,0.835,0.872,0.898,0.893,0.805,0.846,0.835,0.696],
        "std":  [0.082,0.081,0.102,0.083,0.076,0.077,0.095,0.083,0.089,0.108]
    },
    "TPR": {
        "mean": [0.831,0.766,0.734,0.761,0.821,0.828,0.566,0.768,0.761,0.706],
        "std":  [0.132,0.141,0.157,0.131,0.133,0.116,0.152,0.133,0.150,0.161]
    },
    "FPR": {
        "mean": [0.232,0.193,0.203,0.229,0.204,0.226,0.094,0.228,0.241,0.315],
        "std":  [0.137,0.126,0.139,0.142,0.137,0.138,0.099,0.139,0.129,0.149]
    },
    "Cohen's Kappa": {
        "mean": [0.599,0.572,0.530,0.531,0.617,0.603,0.469,0.539,0.519,0.391],
        "std":  [0.185,0.187,0.215,0.185,0.180,0.181,0.183,0.194,0.185,0.216]
    },
    "MCC": {
        "mean": [0.612,0.584,0.542,0.542,0.629,0.612,0.507,0.549,0.531,0.400],
        "std":  [0.186,0.188,0.217,0.187,0.179,0.180,0.183,0.194,0.186,0.219]
    }
}

# ===============================
# Run RCV ranking
# ===============================
plot_metrics_and_save(
    classifiers,
    rcv_metrics,
    "RCV Base-Level Ranking (Std / Mean)"
)
